import numpy as np
import ase.io
from scipy.optimize import minimize, minimize_scalar, OptimizeResult
from typing import Callable, Dict, Sequence, Optional, List
import torch
from torch import FloatTensor, IntTensor
from ..ff.graph import GRAPH_BUILDER_TYPE
from ..ff.baseFF import baseFF
from .calculator import get_ase_atoms
from .geomopt import GeometryFitting
from geomatry.ff.springs import SpringFF
from scipy.stats import chi2
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

def validate_lj_geometry(Ra: FloatTensor, min_nearest: float=0.65, max_nearest: float=1.8) -> None:
    if Ra.shape[0] < 2:
        return
    distances = torch.cdist(Ra.detach(), Ra.detach())
    distances = distances + torch.eye(Ra.shape[0], device=Ra.device, dtype=Ra.dtype) * 1e6
    nearest = distances.min(dim=1).values
    min_dist = nearest.min().item()
    max_dist = nearest.max().item()
    if min_dist < min_nearest or max_dist > max_nearest:
        raise ValueError(
            f"Rejected LJ target geometry: nearest-neighbor distances range "
            f"from {min_dist:.6f} to {max_dist:.6f}; expected [{min_nearest}, {max_nearest}]."
        )

def getMaxDirection(optimizer, params_0, index):
    Ra_star_0=optimizer.fitting._get_Ra_star(params_0,optimizer.graph_builders[index],optimizer.Ra_stars[index],optimizer.Zas[index],optimizer.fixed_atom_indices)
    direction=(Ra_star_0-optimizer.Ra_stars[index]).detach().numpy()
    return direction/np.linalg.norm(direction)

def drawGraphs(optimizer_Energy,params,direction, index):
    if (np.isnan(direction[0,0])):
        direction=np.zeros(direction.shape)
        direction[-1,0]=1
    if ('k' in params.keys()):
        optimizer_Energy.ff.reset_parameters(params['k'],params['r0'])
    elif ('epsilon' in params.keys()):
        optimizer_Energy.ff.reset_parameters(params['epsilon'], params['sigma'])
    else:
        optimizer_Energy.ff.reset_parameters(params)
    Ra_temp=copy.deepcopy(optimizer_Energy.Ra_stars[index])
    atoms=get_ase_atoms(Ra_temp, optimizer_Energy.Zas[index])
    x=np.linspace(-2,2,30)
    z1=copy.deepcopy(x)
    for i in range(len(x)):
        Ra_temp[:,:]+=x[i]*direction
        E0=optimizer_Energy.ff.get_E(Ra_temp,optimizer_Energy.Zas[index],*optimizer_Energy.graph_builders[index](atoms))
        z1[i]=E0
        Ra_temp[:,:]-=x[i]*direction
    E0=optimizer_Energy.ff.get_E(Ra_temp,optimizer_Energy.Zas[index],*optimizer_Energy.graph_builders[index](atoms)).detach()
    plt.scatter(x,z1)
    #plt.scatter(x,E0+x**2)


def truncated_chi2_rvs(df, low, high, size=1):
    cdf_low = chi2.cdf(low, df)
    cdf_high = chi2.cdf(high, df)
    
    # 2. Generate uniform random numbers in the range [cdf_low, cdf_high]
    uniform_variates = np.random.uniform(cdf_low, cdf_high, size)
    
    # 3. Apply the inverse CDF (PPF) to transform the uniform variates
    truncated_variates = chi2.ppf(uniform_variates, df)
    
    return truncated_variates

def callback(intermediate_result: OptimizeResult):
    print("this is a callback")

class SingleSystemOptimizer:
    def __init__(self, 
        Ra: FloatTensor, Za: IntTensor, graph_builder: GRAPH_BUILDER_TYPE,
        ff: baseFF, loss_function: Callable[[FloatTensor, FloatTensor], FloatTensor], 
        Ra_star: Optional[FloatTensor]=None,
        params_star: Optional[Dict[str, FloatTensor]]=None,
        fixed_atom_indices: Optional[Sequence[int]]=None,
        device: torch.device=torch.device("cpu"),
        dtype: torch.dtype=torch.float64,
        fmax: float=1e-4
    ):
        self.ff = ff.to(device, dtype)
        self.Za = Za.to(device, dtype)
        self.graph_builder = graph_builder
        self.fixed_atom_indices = fixed_atom_indices
        self.fitting = GeometryFitting(self.ff, loss_function, fmax)
        self.device = device
        self.dtype = dtype
        if Ra_star is None:
            if params_star is None:
                raise ValueError("Either Ra_star or params_star must be provided")
            self.Ra_star = self.get_Ra_star(Ra, params_star)
            ase.io.write("Ra_star.xyz", get_ase_atoms(self.Ra_star, self.Za))
        else:
            self.Ra_star = Ra_star.to(device, dtype).requires_grad_(False)

    def get_Ra_star(self, Ra: FloatTensor, params_star: Dict[str, FloatTensor]) -> FloatTensor:
        self.fitting.reset_parameters(state_dict=params_star)
        Ra_star = self.fitting._get_Ra_star(params_star, self.graph_builder, Ra, self.Za, self.fixed_atom_indices)
        return Ra_star.detach().clone()

    def objective_function(self, params: np.ndarray) -> float:
        flattened_params = torch.from_numpy(params).to(self.device, self.dtype)
        loss, flattened_gradient = self.fitting.get_loss_and_gradient_flat(
            flattened_params, self.graph_builder, 
            self.Ra_star, self.Za, 
            self.fixed_atom_indices
        )
        print("loss:", loss.item())
        return loss.item(), flattened_gradient.detach().cpu().numpy()

    def optimize(self, params_0: Dict[str, FloatTensor], epoch, print_period, e_size, eN, sg) -> FloatTensor:
        for key in params_0.keys():
            params_0[key]=params_0[key].requires_grad_()
        self.ff.reset_parameters(params_0['k'],params_0['r0'])
        print(self.objective_function(self.fitting._flatten(self.ff.state_dict()).detach().numpy())[0])
        #alpha=torch.ones([3*self.Ra_star.shape[0]],device=self.device,requires_grad=True)
        alpha=torch.ones([1],device=self.device,requires_grad=True)
        optimizer = torch.optim.Adam(list(self.ff.parameters())+[alpha])
        for i in tqdm(range(epoch)):            
            loss=0
            trueloss=0
            optimizer.zero_grad()
            N=self.Ra_star.shape[0]
            #deviations_magnitude=e_size*np.sqrt(truncated_chi2_rvs(3*N,0,3*N+sg*(6*N)**0.5,size=eN)).reshape(1,1,eN,1)
            #deviations_direction=np.random.normal(0,e_size,[N,eN,3]).reshape(1,N,eN,3)
            deviations_magnitude=np.random.normal(0,e_size,[eN//2]).reshape(1,1,eN//2,1)
            deviations_magnitude=np.concatenate([deviations_magnitude,-deviations_magnitude],axis=2)
            if (eN%2==1):
                deviations_magnitude=np.concatenate([deviations_magnitude,np.zeros([1,1,1,1])],axis=2)

            
            '''
            deviations_direction=np.random.normal(0,e_size,[N,1,3]).reshape(1,N,1,3)
            deviations_direction[:,self.fixed_atom_indices,:,:]=0
            if (np.min(np.linalg.norm(np.linalg.norm(deviations_direction,axis=3),axis=1))<0.0000001): continue
            deviations_direction/=np.linalg.norm(np.linalg.norm(deviations_direction,axis=3,keepdims=True),axis=1,keepdims=True)
            '''
            
            Ra_star = self.Ra_star.clone().requires_grad_(True)
            E0 = self.ff.get_E(Ra_star, self.Za,*self.graph_builder(get_ase_atoms(self.Ra_star, self.Za)))
            Fa = self.ff.get_Fa(E0, Ra_star)
            if (torch.norm(Fa)==0): break
            Fa[self.fixed_atom_indices,:]=0
            deviations_direction = (Fa / torch.norm(Fa)).detach().numpy().reshape(1,N,1,3)
            deviations=deviations_direction*deviations_magnitude
            Ra_temp = self.Ra_star.clone().reshape(1,-1,1,3).repeat(1,1,eN,1)
            Ra_temp += deviations
            Energies = torch.zeros([eN],dtype=torch.float64)
            for j in range(eN):
                Energies[j] = self.ff.get_E(Ra_temp[0,:,j,:],self.Za,*self.graph_builder(get_ase_atoms(self.Ra_star, self.Za)))
            y=(Energies-E0).detach().numpy()
            x=deviations_magnitude.reshape(-1)**2
            curvature = 1/(np.sum(x*y)/np.sum(x*x))
            if (curvature<0): curvature=-curvature
            alpha=curvature/2/(e_size**2)
            deviations = torch.tensor(deviations_magnitude,device=self.device,requires_grad=False).float().reshape(eN)
            '''
            Emins = torch.min(Energies,dim=0,keepdim=True)[0]
            Emins = Emins.detach()
            logprob = deviations**2/2/e_size/e_size
            #logalpha = torch.log(torch.sum(torch.sum(torch.exp(alpha.reshape(N,1,3))*(deviations_direction_torch**2),axis=2),axis=0))
            #exponentials=torch.exp(logprob-logalpha*(Energies-Emins).detach())
            exponentials=torch.exp(logprob-alpha*(Energies-Emins).detach())
            NormailzationConstant = torch.sum(exponentials,dim=0,keepdim=True)
            p=(exponentials/NormailzationConstant).detach()
            loss+=alpha*(torch.sum(torch.sum(Energies,axis=0)/Energies.shape[0]-torch.sum(p*Energies,axis=0)))
            #loss+=torch.sum(torch.sum(logalpha*Energies,axis=0)/Energies.shape[0]-torch.sum(p*logalpha*Energies,axis=0))
            trueloss+=-np.sum((-logprob+torch.log(p)).detach().cpu().numpy())
            '''
            loss += torch.sum((deviations.reshape(-1)**2/2/e_size/e_size - alpha * (Energies - E0))**2)/np.sum(deviations_magnitude**2)
            trueloss += torch.sum((deviations.reshape(-1)**2/2/e_size/e_size - alpha * (Energies - E0))**2)/np.sum(deviations_magnitude**2)
            
            
            #deviations_magnitude=((1+np.random.rand(1,eN,1)*(3**3-1))**(1/3))*e_size*((3*N+sg*(6*N)**0.5)**0.5)
            deviations_magnitude=(1+np.random.rand(1,eN,1))*e_size*3
            maxerr=0
            deviations=deviations_direction*deviations_magnitude
            Ra_temp = self.Ra_star.clone().reshape(1,-1,1,3).repeat(1,1,eN,1)
            Ra_temp += deviations
            Energies = torch.zeros([eN],dtype=torch.float64)
            for j in range(eN):
                Energies[j] = self.ff.get_E(Ra_temp[0,:,j,:],self.Za,*self.graph_builder(get_ase_atoms(self.Ra_star, self.Za)))
                dE = -Energies[j] + E0
                maxerr=max(maxerr,torch.max(dE))

            '''
            alpha_neg = 1 / (maxerr+0.001)
            
            deviations = torch.tensor(deviations_magnitude,device=self.device,requires_grad=False).float().reshape(eN)
            for j in range(eN):
                exponentials=torch.exp(-alpha_neg*(Energies-E0).detach())
                p=(1/(1+exponentials)).detach()
                loss+=-alpha_neg*torch.sum((Energies-E0)*(1-p)*((deviations/e_size)**2),axis=0)/Energies.shape[0]
                trueloss+=-np.sum((torch.log(p)).detach().cpu().numpy())
            '''
            loss.backward()
            optimizer.step()
            if (i%print_period==0):
                print(i, alpha, trueloss, maxerr)
                drawGraphs(self,self.ff.state_dict(),deviations_direction.reshape(N,3))
                plt.show()
                '''
                n2=getMaxDirection(self,self.ff.state_dict())
                drawGraphs(self,self.ff.state_dict(),n2)
                plt.show()
                '''
                
                try:
                    self.objective_function(self.fitting._flatten(self.ff.state_dict()).detach().numpy())
                except:
                    print('Error finding minimum')
                
        params_0=copy.deepcopy(self.ff.state_dict())
        print('final',params_0)
        return params_0
    
class MultipleSystemOptimizer:
    def __init__(self, 
        Ras: List[FloatTensor], Zas: List[IntTensor], graph_builders: GRAPH_BUILDER_TYPE,
        ff: baseFF, loss_function: Callable[[FloatTensor, FloatTensor], FloatTensor], 
        Ra_stars: Optional[List[FloatTensor]]=None,
        params_star: Optional[Dict[str, FloatTensor]]=None,
        fixed_atom_indices: Optional[Sequence[int]]=None,
        device: torch.device=torch.device("cpu"),
        dtype: torch.dtype=torch.float64,
        fmax: float=1e-4, reoptimize=True
    ):
        self.ff = ff.to(device, dtype)
        self.Zas = Zas
        for i in range(len(self.Zas)):
            self.Zas[i] = self.Zas[i].to(device=device, dtype=torch.long)
        self.graph_builders = graph_builders
        self.fixed_atom_indices = fixed_atom_indices
        self.fitting = GeometryFitting(self.ff, loss_function, fmax)
        self.device = device
        self.dtype = dtype
        if Ra_stars is None:
            if params_star is None:
                raise ValueError("Either Ra_star or params_star must be provided")
            if reoptimize:
                self.Ra_stars=[self.get_Ra_star(graph_builder, Ra, Za, params_star) for graph_builder, Ra, Za in zip(self.graph_builders, Ras, Zas)]
            else:
                self.Ra_stars=[Ra.to(device, dtype) for Ra in Ras]
            for i, (Ra_star, Za) in enumerate(zip(self.Ra_stars, Zas)):
                if hasattr(self.ff, 'epsilon') and hasattr(self.ff, 'sigma'):
                    validate_lj_geometry(Ra_star)
                ase.io.write(f"Ra_star_{i}.xyz", get_ase_atoms(Ra_star, Za))
        else:
            self.Ra_stars = Ra_stars
            for i in range(len(self.Ra_stars)):
                self.Ra_stars[i] = self.Ra_stars[i].to(device, dtype).requires_grad_(False)

    def get_Ra_star(self, graph_builder, Ra: FloatTensor, Za: IntTensor, params_star: Dict[str, FloatTensor]) -> FloatTensor:
        self.fitting.reset_parameters(state_dict=params_star)
        Ra_star = self.fitting._get_Ra_star(params_star, graph_builder, Ra, Za, self.fixed_atom_indices)
        return Ra_star.detach().clone()

    def objective_function(self, params: np.ndarray, index: int) -> float:
        flattened_params = torch.from_numpy(params).to(self.device, self.dtype)
        loss, flattened_gradient = self.fitting.get_loss_and_gradient_flat(
            flattened_params, self.graph_builders[index], 
            self.Ra_stars[index], self.Zas[index], 
            self.fixed_atom_indices
        )
        print("loss:", loss.item())
        return loss.item(), flattened_gradient.detach().cpu().numpy()

    def optimize(self, params_0: Dict[str, FloatTensor], epoch, print_period, e_size, eN, sg) -> FloatTensor:
        for key in params_0.keys():
            params_0[key]=params_0[key].requires_grad_(True)
        if ('k' in params_0.keys()):
            self.ff.reset_parameters(params_0['k'],params_0['r0'])
        else:
            self.ff.reset_parameters(params_0)
        index=0
        #print(self.objective_function(self.fitting._flatten(self.ff.state_dict()).detach().numpy(),index)[0])
        optimizer = torch.optim.Adam(list(self.ff.parameters()))
        for i in tqdm(range(epoch)):       
            optimizer.zero_grad()     
            loss=0
            trueloss=0
            '''           
            distances=torch.linspace(-1,1,10,dtype=torch.float32).reshape(-1)
            loss+=torch.sum((self.ff.nets['1_1'](distances+4)-(distances)**2)**2)
            loss.backward()
            optimizer.step()
            if (i%print_period==0):
                print(i, loss)
                plt.scatter(distances.detach().numpy(),self.ff.nets['1_1'](distances+4).detach().numpy())
                plt.scatter(distances.detach().numpy(),(distances.detach().numpy())**2)
                plt.show()

            #print(loss,self.ff.nets['1_1'](distances+3))
            continue
            '''
            for temp in range(3):
                index=np.random.randint(0,len(self.Ra_stars))
                N=self.Ra_stars[index].shape[0]
                deviations_magnitude=np.random.normal(0,e_size,[eN//2]).reshape(1,1,eN//2,1)
                deviations_magnitude=np.concatenate([deviations_magnitude,-deviations_magnitude],axis=2)
                #deviations_magnitude=np.linspace(-e_size*2,e_size*2,eN,dtype=np.float32).reshape(1,1,eN,1)
                if (eN%2==1):
                    deviations_magnitude=np.concatenate([deviations_magnitude,np.zeros([1,1,1,1])],axis=2,dtype=np.float32)
                Ra_star = self.Ra_stars[index].clone().requires_grad_(True)
                E0 = self.ff.get_E(Ra_star, self.Zas[index],*self.graph_builders[index](get_ase_atoms(self.Ra_stars[index], self.Zas[index])))
                Fa = self.ff.get_Fa(E0, Ra_star)
                if (torch.norm(Fa)==0): break
                Fa[self.fixed_atom_indices,:]=0
                deviations_direction = (Fa / torch.norm(Fa)).detach().numpy().reshape(1,N,1,3)
                deviations = deviations_direction*deviations_magnitude
                Ra_temp = self.Ra_stars[index].clone().reshape(1,-1,1,3).repeat(1,1,eN,1)
                Ra_temp += deviations
                Energies = torch.zeros([eN],dtype=torch.float64)
                distances=[]
                for j in range(eN):
                    distances.append(torch.sum((Ra_temp[0,0,j,:]-Ra_temp[0,1,j,:]).detach()**2)**0.5)
                    Energies[j] = self.ff.get_E(Ra_temp[0,:,j,:],self.Zas[index],*self.graph_builders[index](get_ase_atoms(self.Ra_stars[index], self.Zas[index])))
                deviations = torch.tensor(deviations_magnitude,device=self.device,requires_grad=False).float().reshape(eN)
                y=(Energies-E0)
                x=deviations.reshape(-1)**2
                
                curvature = (torch.sum(x*y)/torch.sum(x*x)).item()
                alpha = curvature
                if (alpha<5): alpha=5
                alpha=1/(alpha)/2
                #Threshold = 10
                nearMinimumLoss = torch.sum((deviations.reshape(-1)**2/2 - alpha * (Energies - E0))**2)/np.sum(deviations_magnitude**2)/e_size/e_size
                #negativeCurvatureLoss = (Threshold>curvature)*(Threshold-curvature)**2
                #mixedLoss = (negativeCurvatureLoss + nearMinimumLoss*np.exp(curvature-Threshold/2))/(1+np.exp(curvature-Threshold/2))
                mixedLoss = nearMinimumLoss
                loss     += mixedLoss
                trueloss += mixedLoss
            
                #print(torch.norm(Fa)/curvature,curvature,mixedLoss)
                
                
                #deviations_magnitude=((1+np.random.rand(1,eN,1)*(3**3-1))**(1/3))*e_size*((3*N+sg*(6*N)**0.5)**0.5)
                deviations_magnitude=(1+np.random.rand(1,eN,1))*e_size*3
                maxerr=0
                deviations=deviations_direction*deviations_magnitude
                Ra_temp = self.Ra_stars[index].clone().reshape(1,-1,1,3).repeat(1,1,eN,1)
                Ra_temp += deviations
                Energies = torch.zeros([eN],dtype=torch.float64)
                for j in range(eN):
                    Energies[j] = self.ff.get_E(Ra_temp[0,:,j,:],self.Zas[index],*self.graph_builders[index](get_ase_atoms(self.Ra_stars[index], self.Zas[index])))
                    dE = -Energies[j] + E0
                    maxerr=max(maxerr,torch.max(dE))
                epsilon=0.001
                alphaneg=1/(maxerr+epsilon)
                p=1/(1+torch.exp(alphaneg*(-Energies + E0))).detach()
                loss += alphaneg*torch.sum((1-p)*(-Energies + E0))
                trueloss += -torch.sum(torch.log(p))
            #loss=torch.sum((y-x)**2)
            loss.backward()
            optimizer.step()
            if (i%print_period==0):
                print(i, 'curvature', curvature, nearMinimumLoss.item(), 'distance', (torch.norm(Fa)/curvature).item(), 'loss', trueloss.item())
                #print(i, 'curvature', curvature, negativeCurvatureLoss.item(), nearMinimumLoss.item(), 'distance', (torch.norm(Fa)/curvature).item(), 'loss', trueloss.item())
                print('trueloss',trueloss.item())
                print('loss', loss.item())
                print(Energies - E0)
                drawGraphs(self,self.ff.state_dict(),deviations_direction.reshape(N,3),index)
                plt.show()
                '''
                n2=getMaxDirection(self,self.ff.state_dict())
                drawGraphs(self,self.ff.state_dict(),n2)
                plt.show()
                '''
                '''
                try:
                    self.objective_function(self.fitting._flatten(self.ff.state_dict()).detach().numpy(),index)
                except:
                    print('Error finding minimum')
                '''
        params_0=copy.deepcopy(self.ff.state_dict())
        return params_0

class DirectionalImplicitMultipleSystemOptimizer(MultipleSystemOptimizer):
    """Optimizer using directional implicit gradients along the steepest-force direction."""

    def _directional_curvature(
        self,
        Ra: FloatTensor,
        Za: IntTensor,
        graph_builder: GRAPH_BUILDER_TYPE,
        u: FloatTensor,
        step: float,
        n_points: int,
    ) -> FloatTensor:
        window = max(3.0 * float(step), 1e-3)
        n_points = max(7, int(n_points))
        if n_points % 2 == 0:
            n_points += 1
        Ra_base = Ra.detach()
        u = u.detach()
        atoms = get_ase_atoms(Ra_base, Za)
        graph = graph_builder(atoms)
        with torch.no_grad():
            alphas = torch.linspace(
                -window,
                window,
                n_points,
                device=Ra_base.device,
                dtype=Ra_base.dtype,
            )
            energies = torch.stack([
                self.ff.get_E(Ra_base + alpha * u, Za, *graph)
                for alpha in alphas
            ])
            design = torch.stack([
                alphas ** 2,
                alphas,
                torch.ones_like(alphas),
            ], dim=1)
            coeffs = torch.linalg.lstsq(design, energies).solution
            curvature = 2.0 * coeffs[0]
        return curvature.detach()

    def _long_range_force_loss(
        self,
        R_far_value: float,
        weight: float,
    ):
        if not hasattr(self.ff, "nets"):
            return None

        first_param = next(self.ff.parameters())
        longRange_loss = torch.zeros(
            (),
            device=first_param.device,
            dtype=first_param.dtype,
        )

        for net in self.ff.nets.values():
            for i in range(10):
                R_far = torch.tensor(
                    [R_far_value+i],
                    device=first_param.device,
                    dtype=first_param.dtype,
                    requires_grad=True,
                )
                E_far = net(R_far).sum()
                dE_dR = torch.autograd.grad(
                    E_far,
                    R_far,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                longRange_loss = longRange_loss + dE_dR.pow(2).sum()

        return 0.5 * weight * longRange_loss

    def optimize(
        self,
        params_0: Dict[str, FloatTensor],
        epoch,
        print_period,
        e_size,
        eN,
        sg,
        lr: float = 1e-3,
        longRange: bool = False,
        longRange_weight: float = 1.0,
        R_far_value: float = 10.0,
        batch_size: int = 3,
    ) -> FloatTensor:
        for key in params_0.keys():
            params_0[key] = params_0[key].requires_grad_(True)
        if 'k' in params_0.keys():
            self.ff.reset_parameters(params_0['k'], params_0['r0'])
        elif 'epsilon' in params_0.keys():
            self.ff.reset_parameters(params_0['epsilon'], params_0['sigma'])
        else:
            self.ff.reset_parameters(params_0)

        batch_size = max(1, int(batch_size))
        optimizer = torch.optim.Adam(list(self.ff.parameters()), lr=lr)
        pbar = tqdm(range(epoch))
        for i in pbar:
            optimizer.zero_grad()
            effective_longRange = longRange and hasattr(self.ff, "nets")
            total_scale = 0.0
            longRange_loss = None
            did_directional_update = False
            batch_indices = torch.randperm(len(self.Ra_stars))[:min(batch_size, len(self.Ra_stars))]
            for index_tensor in batch_indices:
                index = int(index_tensor.item())
                Ra_star = self.Ra_stars[index].clone().requires_grad_(True)
                E0 = self.ff.get_E(Ra_star, self.Zas[index], *self.graph_builders[index](get_ase_atoms(self.Ra_stars[index], self.Zas[index])))
                Fa = self.ff.get_Fa(E0, Ra_star)
                if self.fixed_atom_indices is not None:
                    Fa[self.fixed_atom_indices, :] = 0

                normFa = torch.norm(Fa)
                if normFa < 1e-8:
                    continue

                u = Fa.detach() / normFa.detach()
                g_dir = torch.dot(Fa.view(-1), u.view(-1))
                H_uu = self._directional_curvature(Ra_star, self.Zas[index], self.graph_builders[index], u, e_size, eN)
                if H_uu.abs() < 1e-8:
                    continue

                t = -g_dir / H_uu
                scale = t / H_uu
                total_scale += t.abs().item()


                # Important: detach the displacement unless you intentionally want
                # gradients through t and u.
                Ra_eval = (Ra_star - t.detach() * u.detach()).detach().clone().requires_grad_(True)

                # Recompute values at Ra_star + t*u
                E_eval = self.ff.get_E(
                    Ra_eval,
                    self.Zas[index],
                    *self.graph_builders[index](
                        get_ase_atoms(Ra_eval.detach(), self.Zas[index])
                    )
                )

                Fa_eval = self.ff.get_Fa(E_eval, Ra_eval)
                                
                if self.fixed_atom_indices is not None:
                    Fa_eval = Fa_eval.clone()
                    Fa_eval[self.fixed_atom_indices, :] = 0

                # Option A: use the original direction u
                g_dir_eval = torch.dot(Fa_eval.reshape(-1), u.reshape(-1))
                
                H_eval = self._directional_curvature(Ra_eval, self.Zas[index], self.graph_builders[index], u, e_size, eN)
                scale = t / H_eval
                grads = torch.autograd.grad(
                    g_dir_eval,
                    self.ff.parameters(),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                for param, grad in zip(self.ff.parameters(), grads):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad += -grad * scale
                did_directional_update = True

            if effective_longRange:
                longRange_loss = self._long_range_force_loss(
                    R_far_value,
                    longRange_weight,
                )
                if longRange_loss is not None:
                    grads_far = torch.autograd.grad(
                        longRange_loss,
                        self.ff.parameters(),
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    for param, grad in zip(self.ff.parameters(), grads_far):
                        if grad is None:
                            continue
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        param.grad += grad

            if total_scale == 0.0 and not effective_longRange:
                break

            if longRange_loss is not None:
                pbar.set_postfix(
                    longRange_loss=f"{longRange_loss.detach().item():.6e}",
                    total_scale=f"{total_scale:.6e}",
                )
            else:
                pbar.set_postfix(
                    total_scale=f"{total_scale:.6e}",
                )

            if i % print_period == 0:
                if did_directional_update:
                    x = torch.linspace(
                        -0.5, 0.5, 50,
                        device=Ra_star.device,
                        dtype=Ra_star.dtype,
                    )
                    y = []
                    for a in x:
                        Ra_test = Ra_star - a * u

                        atoms = get_ase_atoms(
                            Ra_test.detach().cpu(),
                            self.Zas[index],
                        )

                        graph = self.graph_builders[index](atoms)

                        E = self.ff.get_E(
                            Ra_test,
                            self.Zas[index],
                            *graph,
                        )

                        y.append(E.detach().cpu().item())
                    plt.scatter(x,y)
                    plt.show(block=False)
                    plt.pause(0.001)
                    plt.close()
                    print('deviation', t)
                    print(H_uu.item())
                    print('Fratio', g_dir_eval.detach().item()/g_dir.detach().item())
                if longRange_loss is not None:
                    print('longRange_loss', longRange_loss.detach().item())

                current_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                print(f"Iteration {i}, total_scale: {total_scale:.6e}")
            optimizer.step()
        params_0 = copy.deepcopy(self.ff.state_dict())
        return params_0
import math
class ExactLineSearchDirectionalImplicitMultipleSystemOptimizer(MultipleSystemOptimizer):
    """Optimizer using exact line minimization along the maximum force direction."""

    def _find_line_minimum(
        self,
        Ra_star: FloatTensor,
        Za: IntTensor,
        graph_builder: GRAPH_BUILDER_TYPE,
        u: FloatTensor,
    ):
        atoms = get_ase_atoms(Ra_star, Za)
        graph = graph_builder(atoms)

        def projected_force_at_alpha(alpha: float) -> float:
            Ra_line = (Ra_star + float(alpha) * u).detach().clone().requires_grad_(True)
            E = self.ff.get_E(Ra_line, Za, *graph)
            Fa = self.ff.get_Fa(E, Ra_line)
            F_proj = torch.dot(Fa.view(-1), u.view(-1))
            return float(F_proj.detach().cpu().item())

        # ------------------------------------------------------------
        # Cheap Newton-like guess near alpha = 0
        # ------------------------------------------------------------
        alpha_guess = 0.0

        try:
            Ra0 = Ra_star.detach().clone().requires_grad_(True)
            E0 = self.ff.get_E(Ra0, Za, *graph)
            Fa0 = self.ff.get_Fa(E0, Ra0)
            Hessian0 = self.ff.get_Hessian(Fa0, Ra0)

            F0_proj = torch.dot(Fa0.view(-1), u.view(-1))
            H_uu0 = torch.dot(
                u.view(-1),
                torch.mv(Hessian0, u.view(-1)),
            )

            F0_proj = float(F0_proj.detach().cpu().item())
            H_uu0 = float(H_uu0.detach().cpu().item())

            if abs(H_uu0) > 1e-8:
                alpha_guess = F0_proj / H_uu0

        except Exception:
            alpha_guess = 0.0

        # ------------------------------------------------------------
        # Very cheap local search for nearest F.u = 0
        # ------------------------------------------------------------
        base_step = max(1e-3, abs(alpha_guess) + 1e-3)
        base_step = min(base_step, 1)  # Avoid excessively large steps

        alpha_star = None
        best_alpha = 0.0
        best_abs_f = abs(projected_force_at_alpha(0.0))

        for scale in [0.1, 0.4, 1, 4, 10]:
            step = base_step * scale

            probe_alphas = [
                -step,
                -0.5 * step,
                0.0,
                0.5 * step,
                step,
            ]

            probe_values = []

            for a in probe_alphas:
                try:
                    f = projected_force_at_alpha(a)
                    probe_values.append(f)

                    if abs(f) < best_abs_f:
                        best_abs_f = abs(f)
                        best_alpha = a

                except Exception:
                    probe_values.append(None)

            # Find nearest sign change around alpha = 0
            candidates = []

            for i in range(len(probe_alphas) - 1):
                a1 = probe_alphas[i]
                a2 = probe_alphas[i + 1]
                f1 = probe_values[i]
                f2 = probe_values[i + 1]

                if f1 is None or f2 is None:
                    continue

                if f1 == 0.0:
                    candidates.append(a1)

                elif f1 * f2 < 0:
                    # Linear interpolation root estimate
                    alpha_root = a1 - f1 * (a2 - a1) / (f2 - f1)
                    candidates.append(alpha_root)

            if candidates:
                alpha_star = min(candidates, key=lambda a: abs(a))
                break

        # If no sign change was found, use the best nearby point
        if alpha_star is None:
            alpha_star = best_alpha

        # ------------------------------------------------------------
        # Evaluate final geometry and curvature
        # ------------------------------------------------------------
        Ra_min = (Ra_star + alpha_star * u).detach().clone().requires_grad_(True)

        E_min = self.ff.get_E(Ra_min, Za, *graph)
        Fa_min = self.ff.get_Fa(E_min, Ra_min)
        Hessian_min = self.ff.get_Hessian(Fa_min, Ra_min)

        H_uu_min = torch.dot(
            u.view(-1),
            torch.mv(Hessian_min, u.view(-1)),
        )

        return alpha_star, Ra_min.detach(), H_uu_min
    def optimize(
        self,
        params_0: Dict[str, FloatTensor],
        epoch,
        print_period,
        e_size,
        eN,
        sg,
        lr: float = 1e-3,
        longRange: bool = False,
        longRange_weight: float = 1.0,
        R_far_value: float = 10.0,
        batch_size: int = 4,
    ) -> FloatTensor:
        for key in params_0.keys():
            params_0[key] = params_0[key].requires_grad_(True)
        if 'k' in params_0.keys():
            self.ff.reset_parameters(params_0['k'], params_0['r0'])
        elif 'epsilon' in params_0.keys():
            self.ff.reset_parameters(params_0['epsilon'], params_0['sigma'])
        else:
            self.ff.reset_parameters(params_0)

        optimizer = torch.optim.Adam(list(self.ff.parameters()), lr=lr)
        pbar = tqdm(range(epoch))
        for i in pbar:
            optimizer.zero_grad()
            effective_longRange = longRange and hasattr(self.ff, "nets")
            total_scale = 0.0
            alpha_star = 0.0
            index = int(torch.randint(0, len(self.Ra_stars), (1,)).item())
            Ra_star = self.Ra_stars[index].clone().requires_grad_(True)
            E0 = self.ff.get_E(Ra_star, self.Zas[index], *self.graph_builders[index](get_ase_atoms(self.Ra_stars[index], self.Zas[index])))
            Fa = self.ff.get_Fa(E0, Ra_star)
            if self.fixed_atom_indices is not None:
                Fa[self.fixed_atom_indices, :] = 0

            normFa = torch.norm(Fa)
            if normFa < 1e-8:
                break

            u = Fa.detach() / normFa.detach()
            if not effective_longRange:
                batch_indices = torch.randperm(len(self.Ra_stars))[:min(batch_size, len(self.Ra_stars))]
                for index_tensor in batch_indices:
                    index = int(index_tensor.item())
                    Ra_star = self.Ra_stars[index].clone().requires_grad_(True)
                    E0 = self.ff.get_E(Ra_star, self.Zas[index], *self.graph_builders[index](get_ase_atoms(self.Ra_stars[index], self.Zas[index])))
                    Fa = self.ff.get_Fa(E0, Ra_star)
                    if self.fixed_atom_indices is not None:
                        Fa[self.fixed_atom_indices, :] = 0

                    normFa = torch.norm(Fa)
                    if normFa < 1e-8:
                        continue

                    u = Fa.detach() / normFa.detach()
                    alpha_star, Ra_min, H_uu_min = self._find_line_minimum(
                        Ra_star, self.Zas[index], self.graph_builders[index], u
                    )
                    if H_uu_min.abs() < 1e-8:
                        continue

                    Ra_min = Ra_min.detach().requires_grad_(True)
                    E_min = self.ff.get_E(Ra_min, self.Zas[index], *self.graph_builders[index](get_ase_atoms(self.Ra_stars[index], self.Zas[index])))
                    Fa_min = self.ff.get_Fa(E_min, Ra_min)
                    g_dir = torch.dot(Fa_min.view(-1), u.view(-1))
                    scale = alpha_star / H_uu_min
                    total_scale += scale.abs().item()

                    grads = torch.autograd.grad(g_dir, self.ff.parameters(), retain_graph=False, create_graph=False, allow_unused=True)
                    for param, grad in zip(self.ff.parameters(), grads):
                        if grad is None:
                            continue
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        param.grad += grad * scale
                if total_scale == 0.0:
                    break

            longRange_loss = None
            if effective_longRange:
                first_param = next(self.ff.parameters())
                longRange_loss = torch.zeros(
                    (),
                    device=first_param.device,
                    dtype=first_param.dtype,
                )

                for net in self.ff.nets.values():
                    R_far = torch.tensor(
                        [R_far_value],
                        device=first_param.device,
                        dtype=first_param.dtype,
                        requires_grad=True,
                    )
                    E_far = net(R_far).sum()

                    dE_dR = torch.autograd.grad(
                        E_far,
                        R_far,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    longRange_loss = longRange_loss + dE_dR.pow(2).sum()

                longRange_loss = 0.5 * longRange_weight * longRange_loss

                grads_far = torch.autograd.grad(
                    longRange_loss,
                    self.ff.parameters(),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                for param, grad in zip(self.ff.parameters(), grads_far):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad += grad * longRange_weight


            optimizer.step()
            min_epsilon = float(self.ff.epsilon.min()) if hasattr(self.ff, 'epsilon') else None
            if not effective_longRange:
                pbar.set_postfix(alpha_star=f"{alpha_star:.4e}", total_scale=f"{total_scale:.6e}", min_epsilon=f"{min_epsilon:.4e}" if min_epsilon is not None else "N/A")
            elif longRange_loss is not None:
                pbar.set_postfix(longRange_loss=f"{longRange_loss.detach().item():.6e}", min_epsilon=f"{min_epsilon:.4e}" if min_epsilon is not None else "N/A")
            if i % print_period == 0:
                if effective_longRange and longRange_loss is not None:
                    print(longRange_loss.item())
                x = torch.linspace(
                    -2, 2, 100,
                    device=Ra_star.device,
                    dtype=Ra_star.dtype,
                )
                y = []
                for a in x:
                    Ra_test = Ra_star - a * u

                    atoms = get_ase_atoms(
                        Ra_test.detach().cpu(),
                        self.Zas[index],
                    )

                    graph = self.graph_builders[index](atoms)

                    E = self.ff.get_E(
                        Ra_test,
                        self.Zas[index],
                        *graph,
                    )

                    y.append(E.detach().cpu().item())
                plt.scatter(x,y)
                ymin=min(y)-1
                ymax=min(max(y),ymin+10)
                plt.ylim([ymin,ymax])
                plt.show(block=False)
                plt.pause(0.001)
                plt.close()
                print(alpha_star)


                if min_epsilon is not None:
                    print(f"Iteration {i}, alpha_star={alpha_star:.4e}, total_scale={total_scale:.6e}, min_epsilon={min_epsilon:.4e}")
                else:
                    print(f"Iteration {i}, alpha_star={alpha_star:.4e}, total_scale={total_scale:.6e}")

        params_0 = copy.deepcopy(self.ff.state_dict())
        return params_0

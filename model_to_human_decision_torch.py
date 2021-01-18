import sys, pickle, os, pathlib
from collections import defaultdict

import torch

# a couple of useful math operations adjusted for numerical stabilitiy

# this is mathemathically eqivalent to torch.log(1+torch.exp(x)) but it doesn't overflow when x is big.
log_1_plus_exp_x=lambda x: torch.logsumexp(torch.stack((x,torch.zeros_like(x)),dim=-1),dim=1)

# this is mathemathically equivalent to log(1/(1+exp(x)))
log_p=lambda x: -log_1_plus_exp_x(x) 

# this is mathemathically equivalent to log(1 - 1/(1+exp(x)) )
log_1_minus_p=lambda x: x+log_p(x)

# demo:
# x=torch.tensor([-100.0,50.0,-1.0,0.0,1.0,50.0,100.0])
# print('log(1+exp(x)) | direct implementation:', torch.log(1+torch.exp(x)))
# print('log(1+exp(x)) | with logsumexp:', log_1_plus_exp_x(x))

# print('log(1/(1+exp(x))) | direct implementation:', torch.log(1/(1+torch.exp(x))))
# print('log(1/(1+exp(x))) | with logsumexp:', log_p(x))

# print('log(1 - 1/(1+exp(x))) | direct implementation:', torch.log(1-1/(1+torch.exp(x))))
# print('log(1 - 1/(1+exp(x))) | with logsumexp:', log_1_minus_p(x))

class ModelToHumanDecision(torch.nn.Module):
        
    def __init__(self,parameters=None,device=None):
        """
        args:
        parameters (dict) parameter names:initial values (e.g. numpy arrays)
        device (str) input to torch.device
        """
        
        super().__init__()
        if device is None:
            device='cpu'        
        self.device=torch.device(device)

        if hasattr(self,'default_parameters'):
            parameter_dict=self.default_parameters()
        else:
            parameter_dict={}
            
        if parameters is not None:
            parameter_dict.update(parameters)
                        
        for par_name, par_value in parameter_dict.items():
            if not isinstance(par_value,torch.Tensor):
                par_value=torch.tensor(par_value,device=self.device,dtype=torch.float64)
            else:
                par_value=par_value.to(self.device).to(torch.float64)
            setattr(self,par_name,torch.nn.Parameter(par_value))

    def forward(self,log_p_sentences1,log_p_sentences2=None):
        """ transform LM sentence probabilities to human choice NLL 
        provide either a single D_designs x S_sentences sentence log-p matrix or two D_designs-long vectors
        """
                
        if log_p_sentences2 is None:            
            assert log_p_sentences1.shape[1] == 2, 'code currently implemented for sentence *pairs*'
            log_p_sentences2=log_p_sentences1[:,1]
            log_p_sentences1=log_p_sentences1[:,0]
            
        log_p1=torch.as_tensor(log_p_sentences1,device=self.device,dtype=torch.float64)
        log_p2=torch.as_tensor(log_p_sentences2,device=self.device,dtype=torch.float64)
        
        return self._f(log_p1,log_p2)
    
    def get_parameters(self):
        numpy_parameters={}
        for par_name, par in dict(self.named_parameters()).items():
            if par.nelement()==1:
                numpy_parameters[par_name]=par.item()
            else:
                numpy_parameters[par_name]=par.detach().cpu().numpy()
        return numpy_parameters
    
    def save(self,path):
        # save pickle with model definitions
        class_name=self.__class__.__name__        
        parameters=self.get_parameters()
        
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump([class_name, parameters], f)
        
class Naive(ModelToHumanDecision):
    """ sentence log-probabilities are not transformed at all """
    def _f(self,log_p1,log_p2):
        gamma = 1.0
        s1a_b = log_p1 - log_p2
        p1=1/(1+torch.exp(-(s1a_b)/gamma))
        p2=1-p1
        choice_NLL=-torch.log(torch.stack([p1,p2],dim=-1))
        return choice_NLL

class NoSquashing(ModelToHumanDecision):
    """ only gamma scaling """
    def default_parameters(self):
        parameters={}
        parameters['gamma']=10.0
        return parameters
    def _f(self,log_p1,log_p2):
        gamma=self.gamma
        s1a_b = log_p1 - log_p2
        p1=1/(1+torch.exp(-(s1a_b)/gamma))
        p2=1-p1
        choice_NLL=-torch.log(torch.stack([p1,p2],dim=-1))
        return choice_NLL

class FixedWidthSquashing(ModelToHumanDecision):
    def default_parameters(self):
        parameters={}
        parameters['squashes']=200.0 # this doesn't work very well - use minimum sentence probability instead
        parameters['gamma']=10.0
        return parameters
    def _f(self,log_p1,log_p2):
        gamma=self.gamma
        squash_threshold=self.squashes
        width=1.0
        log_p1_corrected=width*log_1_plus_exp_x((log_p1+squash_threshold)/width)-squash_threshold
        log_p2_corrected=width*log_1_plus_exp_x((log_p2+squash_threshold)/width)-squash_threshold
        s1a_b = log_p1_corrected - log_p2_corrected
#         p1=1/(1+torch.exp(-(s1a_b)/gamma))
#         p2=1-p1
#         choice_NLL=-torch.log(torch.stack([p1,p2],dim=-1))        
        log_p1=log_p(-s1a_b/gamma)
        log_p2=log_1_minus_p(-s1a_b/gamma)
        choice_NLL=-torch.stack([log_p1,log_p2],dim=-1)
        return choice_NLL

class VariableWidthSquashing(ModelToHumanDecision):
    def default_parameters(self):
        parameters={}
        parameters['squashes']=200.0 # this doesn't work very well - use minimum sentence probability instead
        parameters['gamma']=10.0
        parameters['width']=1.0
        return parameters
    
    def _f(self,log_p1,log_p2):
        gamma=self.gamma
        squash_threshold=self.squashes
        width=torch.nn.Softplus()(self.width)+1e-1
        log_p1_corrected=width*log_1_plus_exp_x((log_p1+squash_threshold)/width)-squash_threshold
        log_p2_corrected=width*log_1_plus_exp_x((log_p2+squash_threshold)/width)-squash_threshold
#         log_p1_corrected=width*torch.log(1+torch.exp((log_p1+squash_threshold)/width))-squash_threshold
#         log_p2_corrected=width*torch.log(1+torch.exp((log_p2+squash_threshold)/width))-squash_threshold
        s1a_b = log_p1_corrected - log_p2_corrected
        log_p1=log_p(-s1a_b/gamma)
        log_p2=log_1_minus_p(-s1a_b/gamma)
        choice_NLL=-torch.stack([log_p1,log_p2],dim=-1)
#         p1=1/(1+torch.exp(-(s1a_b)/gamma))
#         p2=1-p1
#         choice_NLL=-torch.log(torch.stack([p1,p2],dim=-1))
        return choice_NLL
    
class SquashedSoftmax(ModelToHumanDecision):
    def default_parameters(self):
        parameters={}
        parameters['gamma']=10.0
        parameters['squashes']=-10.0
        return parameters
    def _f(self,log_p1,log_p2):
        gamma=self.gamma
        eta=self.squashes
        
        denominator=torch.logsumexp(torch.stack([log_p1/gamma,torch.ones_like(log_p1)*eta,log_p2/gamma,torch.ones_like(log_p2)*eta],dim=-1),dim=-1)
        log_p1 = torch.logsumexp(torch.stack([log_p1/gamma,torch.ones_like(log_p1)*eta],dim=-1),dim=-1)-denominator
        log_p2 = torch.logsumexp(torch.stack([log_p2/gamma,torch.ones_like(log_p2)*eta],dim=-1),dim=-1)-denominator
        choice_NLL=-torch.stack([log_p1,log_p2],dim=-1)
        return choice_NLL

class VariableWidthSigmoid(ModelToHumanDecision):
    def default_parameters(self):
        parameters={}
        parameters['gamma']=10.0
        parameters['squashes']=0.0
        parameters['width']=1.0
        return parameters
    def _f(self,log_p1,log_p2):
        gamma=self.gamma
        x0=-self.squashes
        k=self.width        
        log_p1_corrected=torch.sigmoid((log_p1-x0)/k)
        log_p2_corrected=torch.sigmoid((log_p2-x0)/k)
        s1a_b = log_p1_corrected - log_p2_corrected
        log_p1=log_p(-s1a_b/gamma)
        log_p2=log_1_minus_p(-s1a_b/gamma)
        choice_NLL=-torch.stack([log_p1,log_p2],dim=-1)
        return choice_NLL
    
def get_parameters_across_models(decision_model_ordered_dict):
    """ collect parameters across models """
    parameters=defaultdict(List)
    for decision_model in decision_model_ordered_dict.values():
        for par_key, par_val in decision_model.parameters().items():
            parameters[par_key].append(par_val.item())

def load_decision_model(path,device=None):
    with open(path, 'rb') as f:
        class_name, parameters=pickle.load(f)
    
    # grab class object from current module by string https://stackoverflow.com/a/17960039
    class_obj=getattr(sys.modules[__name__], class_name)
    print('found saved parameters:',parameters)
    return class_obj(parameters=parameters,device=device)
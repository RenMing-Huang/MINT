import math
from typing import List, Optional, Tuple, Union
import torch

class NullCtx:
    """
    @func: 
    """

    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class AmpOptimizer:
    """
    @func: 
    Automatic Mixed Precision, AMP
    @info: 
    Mixed precision training refers to using FP32 as the sovereign weight, 
    using FP16/BF16 to increase the training speed during forward and backward propagation, 
    and finally using FP16/BF16 gradient to update the FP32 sovereign weight during the gradient update phase.
    """
    
    def __init__(
        self,
        mixed_precision: int, 
        optimizer: torch.optim.Optimizer, 
        names: List[str], 
        paras: List[torch.nn.Parameter],
        grad_clip: float, # gradient clipping threshold to prevent gradient explosion
        n_gradient_accumulation: int = 1, # the number of steps of gradient accumulation allows for larger batch training by accumulating gradients of multiple steps
        fabric=None,  # NEW: optional Fabric instance for distributed training
    ):  
        ## params
        self.enable_amp = mixed_precision > 0 # False when mixed_precision=0
        self.using_fp16_rather_bf16 = mixed_precision == 1 # False when mixed_precision=0
        
        ## create an autocast context manager that converts the operation to fp16 or bf16
        if self.enable_amp:
            self.amp_ctx = torch.autocast('cuda', enabled=True, dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16, cache_enabled=True)
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11, growth_interval=1000) if self.using_fp16_rather_bf16 else None # only fp16 needs a scaler
            # in the case of fp16, we also need to define a GradScaler for gradient scaling to avoid value underflow
            # fp16 has a small dynamic range and needs to amplify the gradient to improve calculation accuracy
        else:
            self.amp_ctx = NullCtx()
            self.scaler = None
        
        ## optimizer
        self.optimizer, self.names, self.paras = optimizer, names, paras   # paras have been filtered so everyone requires grad
        self.fabric = fabric  # Store fabric reference for backward

        ## gradient clip
        self.grad_clip = grad_clip
        self.early_clipping = self.grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm') # crop before scaling | not have global_grad_norm
        self.late_clipping = self.grad_clip > 0 and hasattr(optimizer, 'global_grad_norm') # crop after scaling | have global_grad_norm
        
        ##
        self.r_accu = 1 / n_gradient_accumulation   # scale losses as they are backpropagated according to "n_gradient_accumulation"
        
    def backward_clip_step(
        self, 
        stepping: bool, 
        loss: torch.Tensor,
    ) -> Tuple[Optional[Union[torch.Tensor, float]], Optional[float]]:
        """
        @func: 
        returns a state dictionary for the optimizer and scaler
        """

        ### backward
        loss = loss.mul(self.r_accu) # r_accu == 1.0 / n_gradient_accumulation
        orig_norm = scaler_sc = None
        
        # Use fabric.backward if available, otherwise fallback to loss.backward
        if self.fabric is not None:
            # When using Fabric, it handles the backward pass
            self.fabric.backward(loss, retain_graph=False)
        elif self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=False, create_graph=False)
        else:
            loss.backward(retain_graph=False, create_graph=False)
        
        ### optimizer step
        if stepping:

            ## recover the optimizer
            if self.scaler is not None: 
                self.scaler.unscale_(self.optimizer)

            ## gradient clip
            if self.early_clipping:
                orig_norm = torch.nn.utils.clip_grad_norm_(self.paras, self.grad_clip)
            
            ## scale and step
            if self.scaler is not None:
                
                # the scaler checks for gradient overflow. If there is no overflow, call optimizer.step() to update the parameters; 
                # if an overflow is detected, skip this step to avoid incorrect parameter updates
                self.scaler.step(self.optimizer)
                # get the current scaling factor (the value of the scaler), 
                # which is dynamically adjusted during the gradient calculation to ensure that the gradient neither underflows nor overflows
                scaler_sc: float = self.scaler.get_scale()
                if scaler_sc > 32768.: # fp16 will overflow when > 65536, so multiply 32768 could be dangerous
                    self.scaler.update(new_scale=32768.)
                else:
                    self.scaler.update()
                # monitor the scaler_sc
                try:
                    scaler_sc = float(math.log2(scaler_sc))
                except Exception as e:
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    raise e
            else:
                self.optimizer.step()
            
            ## gradient clip
            if self.late_clipping:
                orig_norm = self.optimizer.global_grad_norm
            
            ## clear all of the grad of params
            self.optimizer.zero_grad(set_to_none=True)
        
        return orig_norm, scaler_sc
    
    def state_dict(self):
        """
        @func: 
        return a state dictionary for the optimizer and scaler
        
        """

        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state, strict=True):
        """
        @func: 
        load the state of the optimizer and scaler and resume the training progress
        
        """

        if self.scaler is not None:
            try: self.scaler.load_state_dict(state['scaler'])
            except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        self.optimizer.load_state_dict(state['optimizer'])



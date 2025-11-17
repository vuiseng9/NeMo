
trainer.fit(model, data)
    call._call_and_handle_interrupt(
            self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
        )
    _fit_impl    
        _run:
            call._call_configure_model(self)
            /opt/venv/lib/python3.12/site-packages/lightning/pytorch/trainer/call.py
            
                    with trainer.profiler.profile(f"[LightningModule]{pl_module.__class__.__name__}.{hook_name}"):
                        output = fn(*args, **kwargs)

                        fn is configure_model
                        /opt/NeMo/nemo/collections/llm/gpt/model/base.py
                        GPTConfig.configure_model
                            model = MCoreGPTModel( import as from GPTModel
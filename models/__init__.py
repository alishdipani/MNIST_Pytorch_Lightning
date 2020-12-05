from .cnn3 import CNN3

def build_model(args):
    if args.model_name=='CNN3':
        return CNN3(args)
    else:
        raise NotImplementedError(f'{args.model_name} not implemented!')
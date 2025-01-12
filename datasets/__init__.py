from .vaw import build as build_vaw
def build_dataset(image_set, args):

    if args.dataset_file == 'vaw':
        return build_vaw(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')

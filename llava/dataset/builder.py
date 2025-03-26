from .video_dataset.builder import build_video_data_modules
from .image_dataset.builder import build_image_data_modules

def build_dataset(data_args, tokenizer):
    if data_args.data_type == "video":
        return build_video_data_modules(tokenizer, data_args)
    elif data_args.data_type == "image":
        return build_image_data_modules(tokenizer, data_args)
    else:
        raise NotImplementedError

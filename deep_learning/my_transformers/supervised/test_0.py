
def get_transforms(train: bool, image_size: int):
    # These transformations are cached.
    # We could have used RandomCrop with padding. But we are smart, and we know we cache the initial_transforms so
    # we don't compute them during runtime. Therefore, we do the padding beforehand, and apply cropping only at
    # runtime
    random_choice = v2.RandomChoice([
        v2.RandomPerspective(
                    distortion_scale=0.15, # controls how much each corner can move. 
                    p=1.0),                # probability of applying the effect
        v2.RandomRotation(degrees=30),     # rotates an image with random angle
        v2.RandomAffine(
                    degrees=30,             # rotation ±30
                    translate=(0.15, 0.15), # horizontal/vertical translation as fraction of image
                    scale=(0.75, 1.05),     # scale factor
                    shear=10),              # shear angle ±10°
        v2.RandomCrop(
                    size=image_size,   # height & width of crop
                    padding=4),        # pixels to pad around the image
        v2.RandomResizedCrop(
                    size=image_size,
                    scale=(0.75, 1.),  # range of area proportion to crop from the original image
                    ratio=(0.8,  1.)), # range of aspect ratio (width/height)
        v2.RandomPosterize(
                    bits=4, # number of bits to keep per channel, -> floor(original_value / 2^(8-bits)) * 2^(8-bits)
                    p=1.),  # probability of applying the transform
        v2.RandomAdjustSharpness(
                    sharpness_factor=1.5, # controls the degree of sharpness; ( >1 sharpened; <1 slightly blurred)
                    p=1.),                      # probability of applying the transform
        v2.RandomAutocontrast(p=1.), # probability of applying the transform
        v2.RandomEqualize( # histogram of pixel values
                    p=1.), # probability of applying the transform
        v2.ColorJitter(  # randomly changes the brightness, contrast, saturation, and hue
                    brightness=0.5, # factor to change brightness
                    contrast=0.3,   # factor to change contrast
                    saturation=0.3, # factor to change saturation
                    hue=0.3,),      # factor to change hue
        v2.GaussianBlur(  # applies a Gaussian blur
                    kernel_size=(7, 7), # size of the Gaussian kernel
                    # standard deviation of the Gaussian kernel; a float or tuple (min, max) for random sampling
                    sigma=(0.1, 5.)),   # how to handle image borders
        v2.JPEG(quality=(5, 70)), # JPEG quality factor (1–100). Lower → more compression artifacts
        v2.AugMix(
                    severity=4,      # severity of base augmentation operators
                    mixture_width=3, # the number of augmentation chains
                    chain_depth=-1,  # number of augmentation operations per chain; -1 → random, 3 → exactly 3
                    # controls how the multiple augmentation chains are mixed;
                    # dirichlet distribution parameter for mixing; 
                    # 0.1 one chain is likely to dominate
                    # 1.0 → default; all 3 chains contribute moderately equally 
                    # 10. -> all chains contribute almost equally; results in smoother, blended augmentations
                    # decrease alpha → more extreme augmentations (one chain dominates)
                    # increase alpha → smoother blending of chains
                    alpha=1.,    
                    all_ops=True,),  # use all operations (including brightness, contrast, color and sharpness)
        v2.RandomErasing(
                    scale=(0.01, 0.15), # range of area ratio to erase (relative to image area)
                    value=10,           # fill value: single number, tuple, or 'random'
                    inplace=False,      # whether to erase in place or return a new image
                    p=1.),              # probability of applying the transform
        v2.Grayscale(num_output_channels=3), # number of channels in output image: 1 or 3
        v2.RandomHorizontalFlip(),
        v2.Identity(),  # returns the input image unchanged
    ])
    # We use the inplace flag because we can safely change the tensors inplace when normalize is used.
    # For is_train=False, we can safely change the tensors inplace because we do it only once, when caching.
    # For is_train=True, we can safely change the tensors inplace because we clone the cached tensors first.

    # Q: How to make this faster?
    # A: Use batched runtime transformations. y
    return v2.Compose(transforms)

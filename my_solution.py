import torch
import torch.utils.data
import torchvision.transforms
import torch.nn
import torch.nn.init
import torch.optim
import torch.autograd

import numpy
import random
import os.path
import shutil
import pandas
import PIL.Image
import time
import matplotlib.pyplot
import tables
import subprocess
import enum
import scipy.ndimage

class Color(enum.Enum):
    red = 0
    green = 1
    blue = 2

class ComposeImageMaskTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform in self.transforms:
            (image, mask) = transform(image, mask)
        return (image, mask)

class ToTensors(object):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()
        
    def __call__(self, image, mask):
        return (self.to_tensor(image), self.to_tensor(mask))

class PadWidths(object):
    def __init__(self, padding):
        self.padding = padding
        
    def __call__(self, image, mask):
        return (self.pad_image_width(image, self.padding), self.pad_image_width(mask, self.padding))
    
    def pad_image_width(self, image, padding):
        (width, height) = image.size
        (left, top, right, bottom) = (-padding, 0, width + padding, height)
        return image.crop((left, top, right, bottom))

class Scales(object):
    def __init__(self, smaller_rescaled_size):
        self.scale = torchvision.transforms.Scale(smaller_rescaled_size)
    
    def __call__(self, image, mask):
        return (self.scale(image), self.scale(mask))
        
class RandomHorizontalFlips(object):
    def __call__(self, image, mask):
        if random.random() < 0.5:
            return (image.transpose(PIL.Image.FLIP_LEFT_RIGHT), mask.transpose(PIL.Image.FLIP_LEFT_RIGHT))
        else:
            return (image, mask)

class RandomRotationsScalesTranslations(object):
    def __call__(self, image, mask):
        self.angle = random.uniform(0, 0)
        self.horizontal_scaling_factor = random.uniform(1, 1)
        self.vertical_scaling_factor = random.uniform(0.9, 1.1)
        self.horizontal_shift_factor = random.uniform(-0, 0)
        self.vertical_shift_factor = random.uniform(-0.1, 0.1)
        self.parameters = [self.angle, self.horizontal_scaling_factor, self.vertical_scaling_factor, self.horizontal_shift_factor, self.vertical_shift_factor]

        return (self.rotate_scale_translate(image, *self.parameters), self.rotate_scale_translate(mask, *self.parameters))
        
    def rotate_scale_translate(self, image, angle, horizontal_scaling_factor, vertical_scaling_factor, horizontal_shift_factor, vertical_shift_factor):
        width = image.size[0]
        height = image.size[1]

        # Create temporary big image with reflected images surrounding original image.
        (temp_width, temp_height) = (3 * width, 3 * height)
        new_image = PIL.Image.new(image.mode, (temp_width, temp_height))
        # First row.
        new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT), (0, 0))
        new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM), (width, 0))
        new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT), (2 * width, 0))
        # Second row.
        new_image.paste(image.transpose(PIL.Image.FLIP_LEFT_RIGHT), (0, height))
        new_image.paste(image, (width, height))
        new_image.paste(image.transpose(PIL.Image.FLIP_LEFT_RIGHT), (2 * width, height))
        # Third row.
        new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT), (0, 2 * height))
        new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM), (width, 2 * height))
        new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT), (2 * width, 2 * height))

        # Rotate, scale, crop centre and resize back to original size.
        new_width = horizontal_scaling_factor * width
        new_height = vertical_scaling_factor * height
        horizontal_shift = horizontal_shift_factor * width
        vertical_shift = vertical_shift_factor * height
        left = ((temp_width - new_width) / 2) + horizontal_shift
        top = ((temp_height - new_height) / 2) + vertical_shift
        right = ((temp_width + new_width) / 2) + horizontal_shift
        bottom = ((temp_height + new_height) / 2) + vertical_shift
        cropping_box = (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))
        new_image = new_image.rotate(angle, resample = PIL.Image.BILINEAR).crop(cropping_box).resize((width, height))

        return new_image
        
class Gaussian(object):
    def __init__(self, height = 1, standard_deviation = 1):
        self.height = height
        self.standard_deviation = standard_deviation

    def __call__(self, x):
        numerator = torch.pow(x, 2)
        denominator = 2 * (self.standard_deviation ** 2)
        return self.height * torch.exp(-numerator / denominator)

class KaggleCarvanaDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, csv_file, image_directory, mask_directory = None, transform = None):
        self.csv_file = csv_file
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.transform = transform

        examples = pandas.read_csv(self.csv_file)
        self.image_names = examples['img']
        self.encoded_masks = examples['rle_mask']
        is_file = lambda image_name: os.path.isfile(self.image_directory + image_name)
        assert self.image_names.apply(is_file).all(), "Some images in " + self.csv_file + " are not found."
        if self.mask_directory is not None:
            is_file = lambda image_name: os.path.isfile(self.mask_directory + image_name.split('.')[-2] + '_mask.gif')
            assert self.image_names.apply(is_file).all(), "Some masks in " + self.csv_file + " are not found."

    def __getitem__(self, index):
        if self.mask_directory is not None:
            labelled_image = PIL.Image.open(self.image_directory + self.image_names[index])
            mask = PIL.Image.open(self.mask_directory + self.image_names[index].split('.')[-2] + '_mask.gif').convert('L')
            if self.transform is not None:
                (labelled_image, mask) = self.transform(labelled_image, mask)
                temp_mask = (mask.numpy().squeeze() * 255).astype('uint8')
                distance = scipy.ndimage.distance_transform_edt(temp_mask) + scipy.ndimage.distance_transform_edt(numpy.invert(temp_mask))
                distance = torch.from_numpy(distance).float()
            return (self.image_names[index], labelled_image, mask, distance)
        else:
            unlabelled_image = PIL.Image.open(self.image_directory + self.image_names[index])
            if self.transform is not None:
                unlabelled_image = self.transform(unlabelled_image)
            return (self.image_names[index], unlabelled_image)
        
    def __len__(self):
        return len(self.image_names.index)

class UNetDown(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, activation = torch.nn.functional.relu, dropout = None):
        super().__init__()
        # Use half-padding so that height and widths of inputs and outputs are the same.
        padding = kernel_size // 2
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding = padding)
        self.activation = activation
        self.dropout = dropout
        
        torch.nn.init.kaiming_normal(self.conv1.weight)
        torch.nn.init.kaiming_normal(self.conv2.weight)

    def forward(self, x):
        x = self.activation(self.conv1(x), inplace = True)
        x = self.activation(self.conv2(x), inplace = True)
        if self.dropout is not None: x = self.dropout(x)
        return x
        
class UNetUp(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, activation = torch.nn.functional.relu):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride = 2)
        padding = kernel_size // 2
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding = padding)
        self.activation = activation
        
        torch.nn.init.kaiming_normal(self.up.weight)
        torch.nn.init.kaiming_normal(self.conv1.weight)
        torch.nn.init.kaiming_normal(self.conv2.weight)

    def forward(self, bridge, x):
        x = self.up(x)
        x = torch.cat((x, bridge), dim = 1)
        x = self.activation(self.conv1(x), inplace = True)
        x = self.activation(self.conv2(x), inplace = True)
        return x

class UNet(torch.nn.Module):
    """
    Based on GPistre's and Heng CherKeng's code from
    https://discuss.pytorch.org/t/unet-implementation/426 and
    https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
    respectively.
    This UNet variation is not the original one in Ronneberger et al.'s paper.
    Main differences are that:
    - Input and output heights and widths are equal by using padded 
    convolutions, whereas the output size is smaller than the input size in the 
    original paper.
    - Initial number of input channels is 3 (RGB) instead of 1.
    - First contracting layer starts from 32 instead of 64 output channels.
    - Depth of this UNet is 6, instead of 5.
    - 1 output channel to be passed through a sigmoid layer, as
    opposed to 2 output channels being passed through a softmax layer in the
    original paper.
    - We dropout whole channels, instead of dropping out with an independent 
    and identical distribution (IID).
    Note that in the original UNet's Caffe implementation:
    - There is an extra ReLU layer directly after every transposed convolution, 
    so there are 3, not 2 ReLU layers per expanding layer. This is not 
    mentioned in the paper, and we do not implement the extra ReLU layer here.
    - The dropout layers occur in the last 2 contracting layers.
    
    """
    def __init__(self, number_of_channels = 3):
        super().__init__()
        self.out_channels = [number_of_channels, 16, 32, 64, 128, 256, 512]
        self.pool = torch.nn.functional.max_pool2d
        
        # Following down and up layers are indexed according to increasing depth, as seen in the UNet architecture in Ronneberger et al.'s paper.
        self.down1 = UNetDown(self.out_channels[0], self.out_channels[1])
        self.down2 = UNetDown(self.out_channels[1], self.out_channels[2])
        self.down3 = UNetDown(self.out_channels[2], self.out_channels[3])
        self.down4 = UNetDown(self.out_channels[3], self.out_channels[4])
        self.down5 = UNetDown(self.out_channels[4], self.out_channels[5], dropout = torch.nn.Dropout2d())
        self.bottom = UNetDown(self.out_channels[5], self.out_channels[6], dropout = torch.nn.Dropout2d())
        
        self.up5 = UNetUp(self.out_channels[6], self.out_channels[5])
        self.up4 = UNetUp(self.out_channels[5], self.out_channels[4])
        self.up3 = UNetUp(self.out_channels[4], self.out_channels[3])
        self.up2 = UNetUp(self.out_channels[3], self.out_channels[2])
        self.up1 = UNetUp(self.out_channels[2], self.out_channels[1])
        self.last = torch.nn.Conv2d(self.out_channels[1], 1, 1)
        
        torch.nn.init.kaiming_normal(self.last.weight)
        
    def forward(self, x):
        feature_map1 = self.down1(x); x = self.pool(feature_map1, 2)
        feature_map2 = self.down2(x); x = self.pool(feature_map2, 2)
        feature_map3 = self.down3(x); x = self.pool(feature_map3, 2)
        feature_map4 = self.down4(x); x = self.pool(feature_map4, 2)
        feature_map5 = self.down5(x); x = self.pool(feature_map5, 2)
        x = self.bottom(x)
        
        x = self.up5(feature_map5, x)
        x = self.up4(feature_map4, x)
        x = self.up3(feature_map3, x)
        x = self.up2(feature_map2, x)
        x = self.up1(feature_map1, x)
        x = self.last(x)
        return x
        
class StableBCELossWithLogits(torch.nn.Module):
    """
    Numerically stable sigmoid binary cross entropy loss that takes in logits, 
    inspired by http://i.imgur.com/jivpFK1.png and based on 
    https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py. This
    is already implemented in 
    http://pytorch.org/docs/master/_modules/torch/nn/functional.html#binary_cross_entropy_with_logits.
    However, we implement it here for the sake of future modification, e.g.
    incorporation of a weight map that both takes into account of border pixels
    and class imbalance.
    """
    def __init__(self, debug = False):
        super().__init__()
        self.debug = debug
     
    def forward(self, logits, targets, weight_maps):
        loss = logits.clamp(min = 0) - (logits * targets) + torch.log(1 + torch.exp(-logits.abs()))
        if self.debug == True:
            print('Unweighted BCE loss = %.3f' % loss.mean().data[0])
        loss = weight_maps * loss
        loss = loss.mean()
        return loss
        
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def get_dice_score(self, logits, targets):
        batch_size = targets.size(0)
        probabilities = torch.nn.functional.sigmoid(logits)
        probabilities = probabilities.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        intersections = probabilities * targets
        
        numerators = 2 * torch.sum(intersections, 1)
        denominators = torch.sum(targets, 1) + torch.sum(probabilities, 1)
        dice_scores = numerators / denominators
        
        return dice_scores.mean()

    def forward(self, logits, targets):
        soft_dice_score = self.get_dice_score(logits, targets)
        # Can try log loss or squared error loss.
        dice_loss = 1 - soft_dice_score
        return dice_loss
        
class BCEAndDiceLoss(torch.nn.Module):
    def __init__(self, debug = False):
        super().__init__()
        self.debug = debug
        self.stable_bce_loss_with_logits = StableBCELossWithLogits(debug = self.debug)
        self.dice_loss = DiceLoss()
        
    def forward(self, logits, targets, weight_maps):
        return self.stable_bce_loss_with_logits(logits, targets, weight_maps) + self.dice_loss(logits, targets)

def print_image(image, output_file_name):
    transform = torchvision.transforms.ToPILImage()
    image = transform(image)
    image.save(output_file_name)
    
def print_blended_image(image, mask, mask_color, output_file_name):
    transform = torchvision.transforms.ToPILImage()
    image = transform(image)
    mask = mask.squeeze()
    temp_mask_shape = ((3, ) + mask.size())
    temp_mask = torch.zeros(temp_mask_shape)
    temp_mask[mask_color.value] = mask
    mask = transform(temp_mask)
    
    blended_image = PIL.Image.blend(image, mask, 0.5)
    blended_image.save(output_file_name)
    
def pad_tensor_width(x, padding):
    (number_of_channels, height, width) = x.shape
    pad = torch.zeros(number_of_channels, height, padding)
    return torch.cat((pad, x, pad), dim = 2)
    
def pad_image_width(image, padding):
    (width, height) = image.size
    (left, top, right, bottom) = (-padding, 0, width + padding, height)
    return image.crop((left, top, right, bottom))
    
def set_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
        
def compute_dice_score(logits, targets, threshold):
    batch_size = targets.size(0)
    probabilities = torch.nn.functional.sigmoid(logits)
    predictions = (probabilities > threshold).float()
    predictions = predictions.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    intersections = predictions * targets
    
    numerators = 2 * torch.sum(intersections, 1)
    denominators = torch.sum(targets, 1) + torch.sum(predictions, 1)
    dice_scores = numerators / denominators
    
    return dice_scores.mean()
    
def cross_validate(cross_validation_loader, threshold = 0.5, print_images = False):
    model.eval()
    number_of_examples = 0
    cross_validation_loss = 0
    dice_score = 0
    for (batch, (image_names, inputs, targets, distances)) in enumerate(cross_validation_loader):
        weight_maps = compute_weight_maps(targets, distances)
        if print_images == True:
            debug_directory = './debugImages/crossValidation/'
            for (i, (image, mask)) in enumerate(zip(inputs, targets)):
                print_image(image, debug_directory + ('image_batch%d_index%d.png' % (batch + 1, i + 1)))
                print_image(mask, debug_directory + ('mask_batch%d_index%d.png' % (batch + 1, i + 1)))
                print_blended_image(image, mask, Color.red, debug_directory + ('blended_batch%d_index%d.png' % (batch + 1, i + 1)))
        
        print('Cross validating batch: %d / %d' % (batch + 1, len(cross_validation_loader)), end = '\r', flush = True)
        # Feed forward.
        batch_size = len(inputs)
        number_of_examples = number_of_examples + batch_size
        inputs = torch.autograd.Variable(inputs, volatile = True).cuda()
        targets = torch.autograd.Variable(targets, volatile = True).cuda()
        weight_maps = torch.autograd.Variable(weight_maps, volatile = True).cuda()
        logits = model(inputs)
        
        # Accumulate dice scores.
        batch_score = compute_dice_score(logits, targets, threshold).data[0]
        dice_score = dice_score + (batch_size * batch_score)
        # Accumulate cross validation batch losses.
        batch_loss = criterion(logits, targets, weight_maps).data[0]
        cross_validation_loss = cross_validation_loss + (batch_size * batch_loss)
    # Average Dice scores and cross validation losses.
    cross_validation_loss = cross_validation_loss / number_of_examples
    dice_score = dice_score / number_of_examples
    print('')

    return (cross_validation_loss, dice_score)

def normalize_to_unity(x):
    return x / x.max()
    
def compute_weight_maps(masks, distances):
    number_of_pixels = masks.size()[-2] * masks.size()[-1]
    number_of_car_pixels = masks.view(masks.size()[0], masks.size()[1], -1).sum(dim = 2)
    car_frequencies = number_of_car_pixels / number_of_pixels
    car_weights = 0.5 / car_frequencies
    car_weights = car_weights.unsqueeze(3).expand_as(masks)
    background_frequencies = 1 - car_frequencies
    background_weights = 0.5 / background_frequencies
    background_weights = background_weights.unsqueeze(3).expand_as(masks)
    class_maps = (masks * car_weights) + ((1 - masks) * background_weights)

    weight_maps = gaussian(distances).unsqueeze(1) + class_maps
    return weight_maps
        
def train_epoch(epoch, train_loader, threshold = 0.5, print_images = False):
    first_batch_loss = 0
    model.train()
    for (batch, (image_names, inputs, targets, distances)) in enumerate(train_loader):
        weight_maps = compute_weight_maps(targets, distances)
        if print_images == True:
            debug_directory = './debugImages/train/'
            for (i, (image_name, image, mask, distance, weight_map)) in enumerate(zip(image_names, inputs, targets, distances, weight_maps)):
                print_image(image, debug_directory + ('image_epoch%d_batch%d_index%d.png' % (epoch + 1, batch + 1, i + 1)))
                print_image(mask, debug_directory + ('mask_epoch%d_batch%d_index%d.png' % (epoch + 1, batch + 1, i + 1)))
                print_blended_image(image, mask, Color.red, debug_directory + ('blended_epoch%d_batch%d_index%d.png' % (epoch + 1, batch + 1, i + 1)))
                print_image(normalize_to_unity(weight_map), debug_directory + ('weight_map_epoch%d_batch%d_index%d.png' % (epoch + 1, batch + 1, i + 1)))
                
        print('Training batch: %d / %d' % (batch + 1, len(train_loader)), end = '\r', flush = True)
        # Feed-forward.
        inputs = torch.autograd.Variable(inputs).cuda()
        targets = torch.autograd.Variable(targets).cuda()
        weight_maps = torch.autograd.Variable(weight_maps).cuda()
        logits = model(inputs)
        
        loss = criterion(logits, targets, weight_maps)
        # Back-propagate and update parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.data[0]
        is_first_batch = (batch == 0)
        if is_first_batch: first_batch_loss = batch_loss

    print('')
    return first_batch_loss
    
def train(number_of_cross_validation_examples, learning_rate_schedule, save_model_interval = 1, debug = False):
    print('Learning rate schedule:', learning_rate_schedule)
    global model
    
    augment = ComposeImageMaskTransforms([
        # Pad image width with 1 pixels at left and right edges to make image size 1280 x 1920 pixels.
        PadWidths(1),
        Scales(height // scale_down_factor),
        RandomRotationsScalesTranslations(),
        ToTensors(),
    ])
    train_set = KaggleCarvanaDataset(mask_csv_file, labelled_image_directory, mask_directory = mask_directory, transform = augment)
    
    # Determine train and cross validation indices, and set up train loader.
    cross_validation_indices = list(range(0, number_of_cross_validation_examples))
    train_indices = list(range(number_of_cross_validation_examples, len(train_set)))
    if debug == True:
        cross_validation_indices = list(range(0, number_of_images_per_car))
        number_of_train_cars = 1
        train_indices = list(range(number_of_images_per_car, (number_of_train_cars + 1) * number_of_images_per_car))
    print('Train indices:', min(train_indices), 'to', max(train_indices))
    print('Cross validation indices:', min(cross_validation_indices), 'to', max(cross_validation_indices))
    (image_name, labelled_image, mask, distance) = train_set[0]
    print('Size of image and mask: %s and %s:' % (tuple(labelled_image.size()), tuple(mask.size())))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 1, sampler = train_sampler, num_workers = 4, pin_memory = True)
    
    # Set up cross validation loader.
    no_augment = ComposeImageMaskTransforms([
        # Pad image width with 1 pixels at left and right edges to make image size 1280 x 1920 pixels.
        PadWidths(1),
        Scales(height // scale_down_factor),
        ToTensors(),
    ])
    cross_validation_set = KaggleCarvanaDataset(mask_csv_file, labelled_image_directory, mask_directory = mask_directory, transform = no_augment)
    cross_validation_loader = torch.utils.data.DataLoader(cross_validation_set, batch_size = 1, sampler = cross_validation_indices, num_workers = 4, pin_memory = True)
    
    # Set up data structures to hold learning curves and F2 scores.
    train_losses = numpy.array([])
    cross_validation_losses = numpy.array([])
    dice_scores = numpy.array([])
    
    # Load previous model and optimizer snapshots, as well as learning curves.
    if start_epoch != 0:
        print('Load snapshot from:', snapshots_directory + 'epoch%d.pth' % start_epoch)
        snapshot = torch.load(snapshots_directory + 'epoch%d.pth' % start_epoch)
        model.load_state_dict(snapshot['state_dict'])
        optimizer.load_state_dict(snapshot['optimizer'])
        train_losses = numpy.load(learning_curves_directory + 'train_losses.npy')[0: start_epoch]
        cross_validation_losses = numpy.load(learning_curves_directory + 'cross_validation_losses.npy')[0: start_epoch]
        dice_scores = numpy.load(learning_curves_directory + 'dice_scores.npy')[0: start_epoch]
    
    for epoch in range(start_epoch, number_of_epochs):
        start_time = time.time()
        # Change to PyTorch's own scheduler class for version 0.2. We can only change the learning rate the following way in version 0.1.
        if epoch in learning_rate_schedule:
            set_learning_rate(optimizer, learning_rate_schedule[epoch])

        # Train and evaluate losses.
        print('Epoch %d: learning rate = %f' % (epoch + 1, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(epoch, train_loader, cross_validation_loader, print_images = False)
        train_losses = numpy.append(train_losses, train_loss)
        print('Epoch %d: train loss = %f' % (epoch + 1, train_loss))
        (cross_validation_loss, dice_score) = cross_validate(cross_validation_loader, threshold = 0.5, print_images = False)
        cross_validation_losses = numpy.append(cross_validation_losses, cross_validation_loss)
        dice_scores = numpy.append(dice_scores, dice_score)
        print('Epoch %d: cross validation loss = %f' % (epoch + 1, cross_validation_loss))
        print('Epoch %d: dice score = %f' % (epoch + 1, dice_score))
        
        # Save learning curves and snapshots.
        numpy.save(learning_curves_directory + 'train_losses.npy', train_losses)
        numpy.save(learning_curves_directory + 'cross_validation_losses.npy', cross_validation_losses)
        numpy.save(learning_curves_directory + 'dice_scores.npy', dice_scores)
        if epoch % save_model_interval == (save_model_interval - 1):
            torch.save({
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
            }, snapshots_directory + 'epoch%d.pth' % (epoch + 1))
        end_time = time.time()
        time_elapsed = end_time - start_time
        print('Epoch %d: time taken = %.2f minutes' % (epoch + 1, time_elapsed / 60))
            
    print('Finished training.')
    print('Train losses:', train_losses)
    print('Cross validation losses:', cross_validation_losses)
    print('Dice scores:', dice_scores)
    torch.save({
        'state_dict': model.state_dict(), 
        'optimizer': optimizer.state_dict(),
    }, snapshots_directory + 'epoch%d.pth' % (epoch + 1))
    
def predict_unlabelled_and_save(dataset_loader, save_file_path, print_images = False):
    hdf5_file = tables.open_file(save_file_path, mode = 'w')
    probabilities_storage = hdf5_file.create_carray(
        hdf5_file.root, 'probabilities',
        tables.UInt8Atom(),
        shape = (len(dataset_loader.sampler), 1, height, width),
        filters = tables.Filters(complevel = 9, complib='blosc'),
    )
    image_names_storage = hdf5_file.create_carray(
        hdf5_file.root, 'image_names',
        # Assigning 32 characters per string should be sufficient, since a typical image name like 0004d4463b50_01.jpg is around 20 characters.
        tables.StringAtom(32),
        shape = (len(dataset_loader.sampler), ),
        filters = tables.Filters(complevel = 9, complib='blosc'),
    )
    
    model.eval()
    # torch.nn.UpsamplingBilinear2d seems to be deprecated for future versions of PyTorch. Use torch.nn.Upsample instead.
    upscale = torch.nn.UpsamplingBilinear2d(scale_factor = scale_down_factor).cuda()    
    start_time = time.time()
    for (batch, (image_names, inputs)) in enumerate(dataset_loader):
        print('Testing and saving batch: %d / %d' % (batch + 1, len(dataset_loader)), end = '\r', flush = True)
        logits = model(torch.autograd.Variable(inputs, volatile = True).cuda())
        probabilities = torch.nn.functional.sigmoid(logits)
        # Remove the 1 pixel left and right borders.
        probabilities = upscale(probabilities).data.cpu()[:, :, :, 1: -1]
        if print_images == True:
            debug_directory = './debugImages/test/'
            for (i, (image, mask)) in enumerate(zip(inputs, probabilities)):
                print_image(image, debug_directory + ('image_batch%d_index%d.png' % (batch + 1, i + 1)))
                print_image(mask, debug_directory + ('mask_batch%d_index%d.png' % (batch + 1, i + 1)))
                
        start = batch * dataset_loader.batch_size
        end = start + len(image_names)
        probabilities_storage[start: end] = (probabilities * 255).numpy().astype('uint8')
        image_names_storage[start: end] = image_names
        hdf5_file.flush()
    hdf5_file.close()
    print('')

    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print('Time taken = %.2f minutes' % (time_elapsed / 60))
    
def rle_encode(mask_image):
    """
    Encodes mask image in run-length encoding (RLE), which is the competition's
    submission format. Written by Sam Stainsby.
    https://www.kaggle.com/stainsby/fast-tested-rle/notebook
    """
    pixels = mask_image.flatten()
    pixels[0] = False
    pixels[-1] = False
    runs = numpy.where(pixels[1: ] != pixels[: -1])[0] + 2
    runs[1: : 2] = runs[1: : 2] - runs[: -1: 2]
    rle_string = ' '.join(str(x) for x in runs)
    
    return rle_string
    
def test(epoch_of_chosen_model, threshold = 0.5, already_predicted = False, debug = False):
    # Save predictions into HDF5 file. Each saved pixel is an uint8 that ranges from 0 to 255.
    save_directory = predictions_directory + 'test/'
    save_file_path = save_directory + 'probabilities.h5'
    if not already_predicted:
        snapshot_path = snapshots_directory + 'epoch%d.pth' % epoch_of_chosen_model
        print('Load snapshot from:', snapshot_path)
        snapshot = torch.load(snapshot_path)
        model.load_state_dict(snapshot['state_dict'])
        downsize = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda image: pad_image_width(image, 1)),
            torchvision.transforms.Scale(height // scale_down_factor),
            torchvision.transforms.ToTensor(),
        ])
        test_set = KaggleCarvanaDataset(sample_submission_csv_file, unlabelled_image_directory, transform = downsize)
        if debug == True:
            test_sampler = list(range(number_of_images_per_car))
        else:
            test_sampler = torch.utils.data.sampler.SequentialSampler(test_set)
        print('Number of test images:', len(test_sampler))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, sampler = test_sampler, num_workers = 4, pin_memory = True)
        predict_unlabelled_and_save(test_loader, save_file_path, print_images = False)
        
    # Write submission file.
    print('Reading predictions from %s.' % save_file_path)
    hdf5_file = tables.open_file(save_file_path, mode = 'r')
    print('Number of images retrieved:', hdf5_file.root.image_names.nrows)
    print('Number of probabilities retrieved:', hdf5_file.root.probabilities.nrows)
    assert hdf5_file.root.image_names.nrows == hdf5_file.root.probabilities.nrows, 'Number of image names is not the same as number of probabilities.'
    with open(submission_file,'w') as f:
        f.write('img,rle_mask\n')
        start_time = time.time()
        for (image_name, probabilities) in zip(hdf5_file.root.image_names, hdf5_file.root.probabilities):
            predictions = probabilities > (threshold * 255)
            f.write(image_name.decode() + ',' + rle_encode(predictions) + '\n')
        end_time = time.time()
        time_elapsed = end_time - start_time
    hdf5_file.close()
    print('Submission file written to %s.' % submission_file)
    print('Time taken = %.2f minutes' % (time_elapsed / 60))
    
def rle_decode(rle_string):
    """
    Decodes run-length-encoded string into 2D mask. Based on Heng CherKeng's 
    code from
    https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
    """
    mask = numpy.zeros((height * width), dtype = numpy.uint8)
    rle_array  = numpy.array([int(s) for s in rle_string.split(' ')]).reshape(-1, 2)
    for white_sequence in rle_array:
        start = white_sequence[0]
        end = start + white_sequence[1]
        mask[start: end] = 255
    mask = mask.reshape(height, width)
    return mask
    
def check_submission():
    submission_file = output_directory + 'submission.csv'
    partial_submission_file = output_directory + 'partial_submission.csv'
    subprocess.call('head --lines 20 %s > %s' % (submission_file, partial_submission_file), shell = True)
    print('Check RLE encoding in sample submission file %s.' % partial_submission_file)
    examples  = pandas.read_csv(partial_submission_file).head(number_of_images_per_car)

    for (i, row) in examples.iterrows():
        image_name = row.values[0]
        rle_mask = row.values[1]
        mask  = rle_decode(rle_mask)
        mask = PIL.Image.fromarray(mask.astype('uint8'), mode = 'L')
        mask.save(predictions_directory + 'test/' + image_name)
    print('Decoded images into %s.' % (predictions_directory + 'test/'))


if __name__ == '__main__':
    mask_csv_file = './data/train_masks.csv'
    sample_submission_csv_file = './data/sample_submission.csv'
    labelled_image_directory = './data/train/'
    unlabelled_image_directory = './data/test/'
    mask_directory = './data/train_masks/'
    (height, width) = (1280, 1918)
    scale_down_factor = 1
    
    # Name your experiment by naming this file.
    experiment_name = os.path.basename(__file__).split('.')[-2]
    output_directory = './results/' + experiment_name + '/'
    template_output_directory = './results/experiment_template/'
    if not os.path.exists(output_directory):
        print('Creating', output_directory)
        shutil.copytree(template_output_directory, output_directory)
    learning_curves_directory = output_directory + 'learningCurves/'
    snapshots_directory = output_directory + 'snapshots/'
    predictions_directory = output_directory + 'predictions/'
    submission_file = output_directory + 'submission.csv'
    number_of_images_per_car = 16
    number_of_cross_validation_cars = 48
    number_of_cross_validation_examples = number_of_cross_validation_cars * number_of_images_per_car
    (start_epoch, number_of_epochs) = (0, 50)
    model = UNet().cuda()
    initial_learning_rate = 3e-4
    optimizer = torch.optim.SGD(model.parameters(), lr = initial_learning_rate, momentum = 0.99, weight_decay = 1e-4)
    criterion = BCEAndDiceLoss(debug = False).cuda()
    gaussian = Gaussian(height = 10, standard_deviation = 10)
    # Dictionary with epochs as keys and learning rates as values.
    learning_rate_schedule = {
        0: initial_learning_rate,
    }
    
    to_train = True
    # Choose epoch for early stopping after examining learning curves.
    epoch_of_chosen_model = None
    if to_train:
        print('Phase: Train')
        only_save_last = number_of_epochs + 1
        train(number_of_cross_validation_examples, learning_rate_schedule, save_model_interval = 5, debug = False)
    if epoch_of_chosen_model is None:
        print('Select epoch of snapshot for loading test model.')
    else:
        print(''); print('Phase: Test')
        test(epoch_of_chosen_model, threshold = 0.5, already_predicted = False, debug = True)
        check_submission()
        # Compress using 7z.
        subprocess.call('7z a %s %s' % (submission_file + '.7z', submission_file), shell = True)
        

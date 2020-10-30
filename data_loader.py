import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

def get_loader(transform,
               mode='train',
               # default batch size
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train', 'valid or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
    """
    path = os.getcwd()
    assert mode in ['train', 'valid', 'test'], "mode must be one of 'train', 'valid' or 'test'."
    if vocab_from_file==False: assert mode=='train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == 'train':
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(path, 'images', 'train2014')
        annotations_file = os.path.join(path, 'annotations', 'captions_train2014.json')
        
    elif mode == 'valid':
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(path, 'images', 'val2014')
        annotations_file = os.path.join(path, 'annotations', 'captions_val2014.json')
        
    elif mode == 'test':
        assert batch_size==1, "Please change batch_size to 1 for testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(path, 'images', 'test2014')
        annotations_file = os.path.join(path, 'annotations', 'image_info_test2014.json')
        

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode == 'train' or mode == 'valid':
        # Randomly sample a caption length and indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        # functionality from torch.utils
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    elif mode == 'test':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        # transform - defined transformation (e.g. Rescale, ToTensor, RandomCrop and etc.)
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        
        # if training and validation
        if self.mode == 'train' or self.mode == 'valid':
            # JSON file, where the annotations are stored
            self.coco = COCO(annotations_file) 
            # each annotatin contains multiple attributes, such as task (e.g. segmentation),
            # image_id, bounding box and etc.
            # in order to load an image, for instance, image URL we will use self.coco.loadImgs(image_id) based on id of image
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            # get all_tokens - a big list of lists. Each is a list of tokens for specific caption
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            # list of token lengths (number of words for each caption)
            self.caption_lengths = [len(token) for token in all_tokens]
            
        else:
            # if we are in testing mode
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]
            
        
    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == 'train':
            # if we are in training mode
            # we retrieve an id of specified annotation
            ann_id = self.ids[index]
            # get caption for annotation based on its id
            caption = self.coco.anns[ann_id]['caption']
            # get image id
            img_id = self.coco.anns[ann_id]['image_id']
            # get an image name, like 'COCO_val2014_000000421535.jpg'
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # Convert image to tensor and pre-process using transform
            # we open specified image and convert it to RGB
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            # specified image transformer - the way we want to augment/modify image
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()
            # return pre-processed image and caption tensors
            # image pre-processed with tranformer applied
            return image, caption

        # obtain image if in test mode
        elif self.mode == 'valid':
            #path = self.paths[index]
            ann_id = self.ids[index]
            # Convert image to tensor and pre-process using transform
            
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)
            
            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()
            # retrun all captions for image (will be required for calculating BLEU score)
            caps_all = []
            # based on image id, get id of all related annotations
            ids_ann = self.coco.getAnnIds(imgIds=img_id)
            for ann_id in ids_ann:
                capt = self.coco.anns[ann_id]['caption']
                caps_all.append(capt)            
            # return original image and pre-processed image tensor
            return image, caption, caps_all
        
        elif self.mode == 'test':
            path = self.paths[index]
            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)
            # return original image and pre-processed image tensor
            return orig_image, image

    def get_train_indices(self):
        # randomly select the caption length from the list of lengths
        sel_length = np.random.choice(self.caption_lengths)
        # retrieve the indices of captions with length as specified above
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        # select m = batch_size captions from list above
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        # return the batch of captions indices
        return indices

    def __len__(self):
        if self.mode == 'train' or self.mode == 'valid':
            return len(self.ids)
        else:
            return len(self.paths)
import json
import os.path as op
from pathlib import Path

from PIL import Image
import numpy as np
from utils.load_files import (
    find_file_path_in_yaml,
    load_box_linelist_file,
    load_from_yaml_file,
)
from utils.logger import LOGGER
from utils.tsv_file import TSVFile
from utils.tsv_file_ops import tsv_reader
import torch
from torchvision import transforms


class VisionLanguageTSVDataset(object):

    def __init__(self,
                 args,
                 yaml_file):

        self.args = args
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.yaml_file = yaml_file
        self.root = Path(op.dirname(yaml_file)).parent.absolute()

        self.cfg = load_from_yaml_file(yaml_file)
        self.cap_linelist_file = find_file_path_in_yaml(
            self.cfg.get('caption_linelist', None), op.join(self.root, 'metadata'))


        self.visual_file = self.cfg.get('img', None)
        self.visual_tsv = self.get_tsv_file(self.visual_file)

        self.label_file = self.cfg.get('label', None)
        self.label_tsv = self.get_tsv_file(op.join('metadata', self.label_file))

        self.cap_file = self.cfg.get('caption', None)
        self.cap_tsv = self.get_tsv_file(op.join('metadata', self.cap_file))

        # True
        if self.cap_linelist_file:
            line_list = load_box_linelist_file(self.cap_linelist_file)
            self.img_line_list = line_list[0]
            self.cap_line_list = line_list[1]
        else:
            # one caption per image/video
            self.img_line_list = [i for i in range(self.label_tsv.num_rows())]
            self.cap_line_list = [0 for i in range(self.label_tsv.num_rows())]


        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()

        self.decoder_num_frames = getattr(args, 'max_num_frames', 2)

        LOGGER.info(f'[PyAV video parameters] '
                    f'Num of Frame: {self.decoder_num_frames} ')


    def roll_func(self, x, axis=1, shift=None, shift_range=50):
        x = torch.as_tensor(x)
        sf = shift
        if shift is None:
            sf = int(np.random.randint(-shift_range, shift_range))

        return x.roll(sf, axis)

    def get_composite_source_idx(self):
        if self.is_composite:
            assert op.isfile(self.cap_linelist_file)
            self.composite_source_idx = [
                int(row[0]) for row in tsv_reader(self.cap_linelist_file)
            ]
        else:
            # only a single tsv file is used as input
            self.composite_source_idx = [
                0 for _ in range(len(self.cap_line_list))
            ]
        return self.composite_source_idx

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)
            return TSVFile(tsv_path)

    def load_caption_to_memory(self):
        self.caption_on_memory = {}
        for img_idx in set(self.img_line_list):
            row = self.get_row_from_tsv(self.cap_tsv, img_idx)
            for cap_idx, data in enumerate(json.loads(row[1])):
                self.caption_on_memory[(img_idx, cap_idx)] = data['caption']

    def prepare_image_keys(self):
        tsv = self.cap_tsv
        return [tsv.get_key(i) for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.cap_tsv
        return {tsv.get_key(i): i for i in range(tsv.num_rows())}

    def get_image_cap_index(self, idx):
        return self.img_line_list[idx], self.cap_line_list[idx]

    def get_row_from_tsv(self, tsv, img_idx):
        row = tsv[img_idx]
        if row[0] != self.image_keys[img_idx]:
            print(row[0], self.image_keys[img_idx])
        assert row[0] == self.image_keys[img_idx]
        return row

    def get_caption(self, img_idx, cap_idx):
        if self.is_train:
            if self.on_memory:
                return self.caption_on_memory[(img_idx, cap_idx)]
            row = self.get_row_from_tsv(self.cap_tsv, img_idx)
            return json.loads(row[1])[cap_idx]['caption']
        return ""

    def get_caption_and_timeinfo(self, data, cap_idx):
        caption, start, end = '', None, None
        data_sample = data[cap_idx]
        caption = data_sample['caption']
        if 'start' in data_sample.keys():
            start = data_sample['start']
        if 'end' in data_sample.keys():
            end = data_sample['end']
        if 'asr' in data_sample.keys() and self.use_asr:
            asr = data_sample['asr'].lower()
        return caption, start, end

    def get_caption_and_timeinfo_wrapper(self, img_idx, cap_idx):
        row = self.get_row_from_tsv(self.cap_tsv, img_idx)
        data_sample = json.loads(row[1])
        caption, start, end = self.get_caption_and_timeinfo(
            data_sample, cap_idx)
        return caption, start, end

    def get_caption_file_in_coco_format(self):
        # for evaluation
        cap_file_coco_format = find_file_path_in_yaml(
            self.cfg.get('caption_coco_format', None), op.join(self.root, 'metadata'))
        if cap_file_coco_format:
            return cap_file_coco_format
        test_split = op.basename(self.yaml_file).split('.')[0]
        return op.join(self.root, 'metadata', test_split + '_caption_coco_format.json')

    def get_captions_by_key(self, key):
        # get a list of captions for image (by key)
        img_idx = self.key2index[key]
        cap_info = json.loads(self.cap_tsv[img_idx][1])
        return [c['caption'] for c in cap_info]

    def convert_string_to_float_array(self, s):
        # Remove brackets and extra characters
        s = str(s)
        
        s = s.replace('[', '').replace(']', '').replace(',', ' ')
        s = s.replace("'", "").replace('"', '')

        # Split the string into individual number strings
        number_strings = s.split()

        # Convert each number string to a float
        float_numbers = [float(num) for num in number_strings]
        float_tensor = torch.tensor(float_numbers, dtype=torch.float32)

        return float_tensor
        #return np.array(float_numbers, dtype=np.float32)

    def get_image(self, bytestring):
        cv2_im = self.convert_string_to_float_array(bytestring)
        return cv2_im

    def get_frames_from_tsv(self, binary_frms):
        # get pre-extracted video frames from tsv files
        frames = []
        #_C, _H, _W = 3, 224, 224                               #CHANGE
        if self.decoder_num_frames > len(binary_frms):
            print(
                f"Corrupt videos, requested {self.decoder_num_frames} frames, "
                f"but got only {len(binary_frms)} frames, will return all zeros instead"
            )

        def sampling(start, end, n):
            if n == 1:
                return [int(round((start + end) / 2.))]
            if n < 1:
                raise Exception("behaviour not defined for n<2")
            step = (end - start) / float(n - 1)
            return [int(round(start + x * step)) for x in range(n)]

        for i in sampling(0, len(binary_frms) - 1, self.decoder_num_frames):
            try:
                image = self.get_image(binary_frms[i])                                                      #CHANGE
            except Exception as e:
                print(f"Corrupt frame at {i}")
                #image = np.zeros((1, _C, _H, _W), dtype=np.int64)
            #_, _C, _H, _W = image.shape                                                 #CHANGE
            frames.append(image)
        return torch.vstack(frames)

    def decode_and_get_frames(self, clip_path_name, start=None, end=None):
        # online decode raw video file, and get video frames
        # output tensor (T, C, H, W), channel is RGB, T = self.decoder_num_frames
        if 'TVC' in clip_path_name:
            # default clip_path_name: datasets/TVC/videos/{tv_show}/{tv_show}_clips/{tv_show}_{seasoninfo}/{video_id}.mp4_{start_time}_{end_time}
            # To load video file, we will need to remove start&end info here
            resolved_video_path = '_'.join(clip_path_name.split('_')[0:-2])
        else:  # VATEX, MSVD, MSRVTT, Youcook2
            resolved_video_path = clip_path_name
        frames, video_max_pts = extract_frames_from_video_path(
            resolved_video_path, self.decoder_target_fps,
            self.decoder_num_frames, self.decoder_multi_thread_decode,
            self.decoder_sampling_strategy, self.decoder_safeguard_duration,
            start, end)
        return frames

    def get_visual_data(self, idx, start=None, end=None):
        row = self.get_row_from_tsv(self.visual_tsv, idx)
        # if the input is a video tsv with frames pre-extracted,
        # return a video-frame tensor
        if len(row) >= self.decoder_num_frames +1:            
            return self.get_frames_from_tsv(row[1:]), True      

    def __len__(self):
        return len(self.img_line_list)

    def __getitem__(self, idx):

        img_idx, cap_idx = self.get_image_cap_index(idx)

        img_key = self.image_keys[img_idx]

        caption_sample, start, end = self.get_caption_and_timeinfo_wrapper(img_idx, cap_idx)

        preproc_frames, is_video = self.get_visual_data(img_idx, start, end)

        # tokenize caption and generate attention maps
        # it will consider only # of visual tokens for building attention maps. # is args.max_img_seq_length
        if isinstance(caption_sample, dict):
            caption = caption_sample["caption"]
        else:
            caption = caption_sample
            caption_sample = None


        return img_key, caption, preproc_frames



class VisionLanguageTSVYamlDataset(VisionLanguageTSVDataset):
    """ TSVDataset taking a Yaml file for easy function call
    """

    def __init__(self,
                 args,
                 yaml_file):
        # print('Init video/image captioning dataloader...')
        super(VisionLanguageTSVYamlDataset,
              self).__init__(args, yaml_file)     


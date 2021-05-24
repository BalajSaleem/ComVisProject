from obj_shadow_detection import get_obj_shadow_masks_predictor
from project_utils import process_lisa_outputs
from project_utils import inpaint
from libs.pconv_model import PConvUnet
import numpy as np

class LinkedModels:
    def __init__(self, lisa_config, pconv_weights):
        self.lisa_predictor = get_obj_shadow_masks_predictor(lisa_config)
        self.pconv_model = PConvUnet(vgg_weights=None, inference_only=True)
        self.pconv_model.load(pconv_weights, train_bn=False)
        self._internal_state = None
        self._internal_state_image = None

    def run_lisa(self, image):
        lisa_outputs, _ = self.lisa_predictor.run_on_image(image)
        self._internal_state = process_lisa_outputs(lisa_outputs)
        self._internal_state_image = image.copy()
        return self._internal_state

    def rm_objs(self, obj_indx, mask_types, dilation=True):
        if not self._internal_state:
            raise Exception('No internal state, call `run_lisa` to set the state')

        assert len(obj_indx) == len(mask_types)

        if self.get_object_no() == 0:
            return self._internal_state_image.copy()

        obj_indx = np.array(obj_indx)
        mask_types = np.array(mask_types)
        
        idx = np.where(obj_indx >= 0 & obj_indx < self.get_object_no())[0]

        obj_indx = obj_indx[idx]
        mask_types = mask_types[idx]

        mask_pairs = self._internal_state[obj_indx]
        masks = [mask_pairs[mask_type] for mask_type in mask_types]

        return inpaint(self.pconv_model, self._internal_state_image, masks, dilation=dilation)

    def get_object_no(self):
        if self._internal_state:
            return len(self._internal_state)
        else:
            return 0

    def get_masks(self):
        if self._internal_state:
            return self._internal_state.copy()
        else:
            return None

    def run_full_pipeline(self, image, rm_objs = [0], mask_types = [1], dilation=True):
        self.run_lisa(image)
        return self.rm_objs(rm_objs, mask_types, dilation)

try:
  import torch as th
  from torchvision.models import mobilenet_v2
except:
  pass

try:
  import tensorflow.compat.v1 as tf
except:
  pass

import os
import numpy as np
import constants as ct

from torchvision import transforms
from collections import OrderedDict
from libraries_pub import LummetryObject

__version__ = '1.0.0.0'

def save_th_mobilenetv2():
  #model prepare
  import torch as th
  from torchvision.models import mobilenet_v2
  model = mobilenet_v2(pretrained=True)
  th.save(model.state_dict(), 'mobilenetv2.th')
  return

class PytorchGraph(LummetryObject):
  def __init__(self, config_graph, **kwargs):
    self.__version__ = __version__
    self.config_graph = config_graph
    self.IS_CUDA_AVAILABLE = th.cuda.is_available()
    self.DEVICE = th.device(ct.CUDA0 if self.IS_CUDA_AVAILABLE else ct.CPU)
    super().__init__(**kwargs)
    return
  
  def startup(self):
    self._load_classes()
    self._load_graph()
    return  

  def _timer_name(self, name):
    return self.__class__.__name__.upper() + '_' + name
  
  def _load_classes(self):
    cls_file = self.config_graph[ct.CLASSES]
    classes = self.log.load_json(os.path.join(self.log.get_models_folder(), cls_file))
    self.classes = {int(k): v[1] for k,v in classes.items()}
    if self.DEBUG:
      self.log.p('Loaded {} classes from {}'.format(len(self.classes), cls_file))
    return
  
  def _load_graph(self):
    if self.DEBUG:
      self.log.p('Loading PYTORCH model', color='g')
    timer_name = self._timer_name(ct.TIMER_LOAD_GRAPH)
    self.log.start_timer(timer_name)
    self.model_size = self.config_graph['MODEL_SIZE']
  
    graph_name = self.config_graph['GRAPH']
    if self.DEBUG:
      self.log.p('Loading graph from models: {}'.format(graph_name))
    self.model = self.log.load_pytorch_model(
      model_name=graph_name, 
      DEBUG=self.DEBUG
      )
    
    path = self.log.get_models_file('mobilenetv2.th')
    model = mobilenet_v2(pretrained=False)
    model.load_state_dict(th.load(path))
    model.to(self.DEVICE)
    
    # device = next(model.parameters()).device
    # self.log.p('Pytorch model running on: {}'.format(device))
    
    self.model = model

    if self.DEBUG:    
      self.log.p('Setting model to {} and eval mode.'.format(self.DEVICE))
    self.model.to(self.DEVICE).eval()
    if self.DEBUG:
      self.log.p('Model on CUDA {}. Mode : {}'.format(
      next(self.model.parameters()).is_cuda, not self.model.training)
      )
    self.log.stop_timer(timer_name)
    return
  
  def _postprocess_inference(self, preds):
    if isinstance(preds, list):
      preds = preds[0]
    np_preds = preds.argmax(axis=-1)
    np_probs = np.take(preds, np_preds)
    np_probs = np.around(np_probs * 100, decimals=2)
    lst_preds = list(zip(np_preds, np_probs))
    lst_out = []
    for idx, proba in lst_preds:
      lbl = idx if not hasattr(self, 'classes') else self.classes[idx]
      if hasattr(self, 'probas'):
        if proba >= self.probas[idx]:
          lst_out.append({ 'PROB_PRC': proba, 'TYPE': lbl })
      else:
        lst_out.append({ 'PROB_PRC': proba, 'TYPE': lbl })
      #end frame iter      
    return lst_out
  
  def _predict(self, images):
    timer_name = self._timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    with th.no_grad():
      if isinstance(images, np.ndarray):
        assert len(images.shape) == 4, 'Tensor should be of form NCHW'
        th_x = th.from_numpy(images).to(self.DEVICE)
      elif isinstance(images, th.Tensor):
        assert len(images.size()) == 4, 'Tensor should be of form NCHW'
        th_x = images
        if self.IS_CUDA_AVAILABLE and not th_x.is_cuda:
          th_x = th_x.to(self.DEVICE)              
      else:
        raise ValueError('Tensor not properly processed!')
      #endif
      
      preds = self.model(th_x).cpu().numpy()
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_images(self, images):
    timer_name = self._timer_name(name=ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    transform = transforms.Compose([
      transforms.Lambda(lambda x: x[:,:,::-1]),
      transforms.ToPILImage(),
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    lst = []
    for image in images:
      th_i = transform(image)
      th_i = th_i.unsqueeze(0)
      lst.append(th_i)
    th_x = th.cat(lst, axis=0)
    self.log.stop_timer(timer_name)
    return th_x
  
  def _run_inference(self, images):
    timer_name = self._timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    preds = self._predict(images)
    preds = self._postprocess_inference(preds)
    self.log.stop_timer(timer_name)
    return preds
  
  def predict(self, np_imgs):
    timer_name = self._timer_name(ct.TIMER_PREDICT)
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()
    
    model_size = self.config_graph.get('MODEL_SIZE')
    self.center_image = model_size is not None and model_size != []
    dct_result = OrderedDict()
    dct_meta = OrderedDict()    
    
    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      images = self._preprocess_images(np_imgs)
      result = self._run_inference(images)
    #endif
    
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result


class TensorflowGraph(LummetryObject):
  def __init__(self, config_graph, **kwargs):
    self.__version__ = __version__
    self.config_graph = config_graph
    self.tf_runoptions = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    super().__init__(**kwargs)
    return
  
  def startup(self):
    self._load_classes()
    self._load_graph()
    return  

  def _timer_name(self, name):
    return self.__class__.__name__.upper() + '_' + name
  
  def _load_classes(self):
    def _set_thresholds():
      if self.DEBUG:
        self.log.p('Setting thresholds')
      model_thr = self.config_graph['MODEL_THRESHOLD']
      dct_class_thr = self.config_graph.get('CLASS_THRESHOLD', {})
      self.probas = {c: dct_class_thr[c] if c in dct_class_thr else model_thr \
                     for c in self.classes.keys()}
      if self.DEBUG:
        self.log.p('Done setting thresholds')
      return
    
    cls_file = self.config_graph['CLASSES']
    full_cls_file = os.path.join(self.log.get_models_folder(), cls_file)
    if self.DEBUG:
      self.log.p('Loading {}...'.format(full_cls_file))
    with open(full_cls_file) as f:
      lines = f.read().splitlines()
    orig_classes = lines.copy()
    self.classes ={i:x for i,x in enumerate(orig_classes)}
    _set_thresholds()
    if self.DEBUG:
      self.log.p('Loaded {} classes from {}'.format(len(lines), cls_file))
    return
  
  def _load_graph(self):
    timer_name = self._timer_name(ct.TIMER_LOAD_GRAPH)
    self.log.start_timer(timer_name)
    graph_name = self.config_graph['GRAPH']
    if self.DEBUG:
      self.log.p('Loading graph from models: {}'.format(graph_name))
    graph = self.log.load_graph_from_models(graph_name)

    assert graph is not None, 'Graph not found!'
    
    self.classes_tensor_name = self.config_graph["CLASSES_TENSOR_NAME"]
    self.scores_tensor_name = self.config_graph["SCORES_TENSOR_NAME"]
    self.boxes_tensor_name = self.config_graph["BOXES_TENSOR_NAME"]
    self.input_tensor_name = self.config_graph["INPUT_TENSOR_NAME"]
    self.numdet_tensor_name = self.config_graph["NUMDET_TENSOR_NAME"]
    
    self.sess = tf.Session(graph=graph)
    self.tf_classes = self.sess.graph.get_tensor_by_name(self.classes_tensor_name+":0")
    self.tf_scores = self.sess.graph.get_tensor_by_name(self.scores_tensor_name+":0")
    self.tf_boxes = self.sess.graph.get_tensor_by_name(self.boxes_tensor_name+":0")
    self.tf_numdet = self.sess.graph.get_tensor_by_name(self.numdet_tensor_name+":0")
    self.tf_input = self.sess.graph.get_tensor_by_name(self.input_tensor_name+":0")
    self.tensors_output = [self.tf_scores, self.tf_boxes, self.tf_classes]
    self.log.stop_timer(timer_name)
    return
  
  def _sess_run(self, images):
    if self.DEBUG:
      self.log.p('Session run PB')
    
    timer_name = self._timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    out_scores, out_boxes, out_classes = self.sess.run(
      self.tensors_output,
      feed_dict={self.tf_input: images},
      options=self.tf_runoptions
      )
    self.log.stop_timer(timer_name)
    return out_scores, out_boxes, out_classes
  
  def _postprocess_boxes(self, boxes, idx_image=0):
    tn = self._timer_name(name=ct.TIMER_POSTPROCESS_BOXES)
    self.log.start_timer(tn)
    img_shape = self.input_shape[idx_image]
    if self.center_image:
      (top, left, bottom, right), (new_h, new_w) = self.log.center_image_coordinates(
        src_h=img_shape[0], 
        src_w=img_shape[1], 
        target_h=self.resize_shape[0], 
        target_w=self.resize_shape[1]
        )
      #[0:1] to [0:yolo_model_shape]
      boxes[:,0] = boxes[:,0] * self.resize_shape[0]
      boxes[:,1] = boxes[:,1] * self.resize_shape[1]
      boxes[:,2] = boxes[:,2] * self.resize_shape[0]
      boxes[:,3] = boxes[:,3] * self.resize_shape[1]
      
      #eliminate centering
      boxes[:,0] = boxes[:,0] - top
      boxes[:,1] = boxes[:,1] - left
      boxes[:,2] = boxes[:,2] - top
      boxes[:,3] = boxes[:,3] - left
      
      #translate to original image
      boxes[:,0] = boxes[:,0] / new_h * img_shape[0]
      boxes[:,1] = boxes[:,1] / new_w * img_shape[1]
      boxes[:,2] = boxes[:,2] / new_h * img_shape[0]
      boxes[:,3] = boxes[:,3] / new_w * img_shape[1]
    else:
      boxes[:,0] = boxes[:,0] * img_shape[0]
      boxes[:,1] = boxes[:,1] * img_shape[1]
      boxes[:,2] = boxes[:,2] * img_shape[0]
      boxes[:,3] = boxes[:,3] * img_shape[1]
    #endif
    boxes = boxes.astype(np.int32)
    boxes[:, 0] = np.maximum(0, boxes[:, 0])
    boxes[:, 1] = np.maximum(0, boxes[:, 1])
    boxes[:, 2] = np.minimum(img_shape[0], boxes[:, 2])
    boxes[:, 3] = np.minimum(img_shape[1], boxes[:, 3])
    self.log.stop_timer(tn)
    return boxes
  
  def _postprocess_inference(self, scores, boxes, classes):
    batch_frames = []
    nr_frames = len(scores)
    for nr_img in range(nr_frames):
      frame_data = []
      frame_scores = scores[nr_img]
      frame_boxes = boxes[nr_img]
      frame_boxes = self._postprocess_boxes(frame_boxes, idx_image=nr_img)
      frame_classes = classes[nr_img].astype(int)
      for _id in range(frame_classes.shape[0]):
        idx_class = frame_classes[_id]
        _type = self.classes[idx_class]
        lst_exclude = self.config_graph.get('EXCLUDE_CLASS', [])
        if _type in lst_exclude:
          continue
        if frame_scores[_id] >= self.probas[idx_class]:
          frame_data.append({
                "TLBR_POS":np.around(frame_boxes[_id]).tolist(), # [TOP, LEFT, BOTTOM, RIGHT]
                "PROB_PRC":np.around(frame_scores[_id] * 100).item(),
                "TYPE": _type
              })
      #end frame iter      
      batch_frames.append(frame_data)
    return batch_frames
  
  def _run_inference(self, images):
    assert images is not None and type(images) == np.ndarray and len(images.shape) == 4
    timer_name = self._timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    scores, boxes, classes = self._sess_run(images)
    preds = self._postprocess_inference(scores, boxes, classes)
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_images(self, np_imgs):
    timer_name = self._timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(np_imgs, (np.ndarray)) and len(np_imgs.shape) == 3:
      np_imgs = np.expand_dims(np_imgs, axis=0)
    
    lst_shape = [x.shape for x in np_imgs]
    self.input_shape = lst_shape
    if len(set(lst_shape)) > 1:
      self.center_image = True
      unique, counts = np.unique(self.input_shape, return_counts=True, axis=0)
      self.resize_shape = tuple(unique[np.argmax(counts)])
      res_h, res_w, _ = self.resize_shape
      lst_centered = [self.log.center_image(x, res_h, res_w) 
                  if x.shape != self.resize_shape else x for x in np_imgs]
      lst_imgs = [x[:,:,::-1] for x in lst_centered]
      np_imgs = np.array(lst_imgs)
    else:
      if type(np_imgs) is list:
        np_imgs = np.array(np_imgs)
    self.log.stop_timer(timer_name)
    return np_imgs
  
  def predict(self, np_imgs):
    timer_name = self._timer_name(ct.TIMER_PREDICT)
    self.log.start_timer(timer_name)
    self.center_image = False
    self.input_shape = None
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()
    
    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result

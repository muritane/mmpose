import time
from typing import List, Union

import os
import rospy
from std_msgs.msg import String
import json

from mmpose.utils.timer import RunningAverage
from ..node import Node
from ..registry import NODES

from mmlab_msgs.msg import PersonArray, Person, Keypoint


@NODES.register_module()
class ModelInferenceROSNode(Node):
    """Publish model inference of PoseTrackerNode as std_msgs/String msg."""

    def __init__(self, name: str, frame_buffer: str, object_buffer: str,
                 topic: str, bbox_info: List[str], keypoint_info: List[str],
                 kp_labels_of_interest: List[str]):
        super().__init__(name=name, enable=True)
        self.synchronous = None

        # Register buffers
        # The trigger buffer depends on the executor.synchronous attribute,
        # so it will be set later after the executor is assigned in
        # ``set_executor``.
        self.register_input_buffer(object_buffer, 'object', trigger=False)
        self.register_input_buffer(frame_buffer, 'frame', trigger=False)
        
        self.bbox_info = bbox_info
        self.keypoint_info = keypoint_info
        self.kp_labels_of_interest = kp_labels_of_interest
        self.keypoint_names = None
        self.keypoint_indices = None

        rospy.init_node(name)
        self.model_inference_pub = rospy.Publisher(topic, PersonArray, queue_size=10)

    def set_executor(self, executor):
        super().set_executor(executor)
        # Set synchronous according to the executor
        if executor.synchronous:
            self.synchronous = True
            trigger = 'object'
        else:
            self.synchronous = False
            trigger = 'frame'

        # Set trigger input buffer according to the synchronous setting
        for buffer_info in self._input_buffers:
            if buffer_info.input_name == trigger:
                buffer_info.trigger = True

    def process(self, input_msgs):
        def get_keypoint_names_and_indices(keypoint_info, kp_labels_of_interest):
            keypoint_names = []
            keypoint_indices = []
            for v in keypoint_info.values():
                keypoint_name = v['name']
                is_label_of_interest = any([label in keypoint_name for label in kp_labels_of_interest])
                if not is_label_of_interest:
                    continue
                keypoint_names.append(keypoint_name)
                keypoint_indices.append(v['id'])
            return keypoint_names, keypoint_indices

        if rospy.is_shutdown():
            print("ROS is shutdown -> exit")
            self._event_manager.set('_exit_')
            return

        object_msg = input_msgs['object']
        objects = [] if object_msg is None else object_msg.get_objects()

        person_array_msg = PersonArray()
        person_array_msg.header.stamp = rospy.Time.now()
        for obj in objects:
            if obj['label'] != 'person':
                continue

            person_msg = Person()
            person_msg.track_id = obj['track_id']
            person_msg.bbox.data = obj['bbox'].tolist()
            person_msg.bbox.info = self.bbox_info

            if self.keypoint_names is None or self.keypoint_indices is None:
                self.keypoint_names, self.keypoint_indices = get_keypoint_names_and_indices(
                        obj['pose_model_cfg'].dataset_info['keypoint_info'], self.kp_labels_of_interest)

            person_msg.keypoints.names = self.keypoint_names
            person_msg.keypoints.info = self.keypoint_info
            keypoints_of_interest = obj['keypoints'][self.keypoint_indices]
            person_msg.keypoints.locations = [Keypoint(kp) for kp in keypoints_of_interest]

            person_array_msg.persons.append(person_msg)

        self.model_inference_pub.publish(person_array_msg)

    def _get_node_info(self):
        info = super()._get_node_info()
        return info


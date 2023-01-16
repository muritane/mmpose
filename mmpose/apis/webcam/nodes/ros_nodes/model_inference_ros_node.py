import time
from typing import List, Union

import os
import rospy
from std_msgs.msg import String
import json

from mmpose.utils.timer import RunningAverage
from ..node import Node
from ..registry import NODES


@NODES.register_module()
class ModelInferenceROSNode(Node):
    """Publish model inference of PoseTrackerNode as std_msgs/String msg."""

    def __init__(self, name: str, frame_buffer: str, object_buffer: str,
                 topic: str):
        super().__init__(name=name, enable=True)
        self.synchronous = None

        # Register buffers
        # The trigger buffer depends on the executor.synchronous attribute,
        # so it will be set later after the executor is assigned in
        # ``set_executor``.
        self.register_input_buffer(object_buffer, 'object', trigger=False)
        self.register_input_buffer(frame_buffer, 'frame', trigger=False)
        
        rospy.init_node(name)
        self.model_inference_pub = rospy.Publisher(topic, String, queue_size=10)

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
        if rospy.is_shutdown():
            print("ROS is shutdown -> exit")
            self._event_manager.set('_exit_')

        

        object_msg = input_msgs['object']
        objects = [] if object_msg is None else object_msg.get_objects()

        labels = [d['label'] for d in objects]
        rospy.loginfo("labels: {}".format(labels))
        if "person" in labels:
            rospy.loginfo([d for d in objects if d['label'] == 'person'])
#            input_keys = [] if input_msgs is None else input_msgs.keys()
#            rospy.loginfo(input_keys)
#            if 'frame' in input_keys:
#                rospy.loginfo(input_msgs['frame'].get_image())
            self._event_manager.set('_exit_')

        # self.model_inference_pub.publish(str(objects))

    def _get_node_info(self):
        info = super()._get_node_info()
        return info


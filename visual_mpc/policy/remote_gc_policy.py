import socket
import struct
import numpy as np

from visual_mpc.policy.policy import Policy

class GCRemotePolicy(Policy):
    
    """
    Uses a remote policy conditioned on goal image
    """

    def __init__(self, agentparams, policyparams, gpu_id, ngpu):
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)
        self.agentparams = agentparams
        self.adim = agentparams['adim']

        self.setup_socket()

    def setup_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self._hp.host, self._hp.port))
        self.socket.setblocking(1) # Set blocking mode to never timeout
        print('Connected to ', self._hp.host, ' port: ', self._hp.port)

    def _receive_data_size(self):
        size = struct.unpack('I', self._receive_bytes(4))[0]
        return size

    def _receive_message_string(self):
        size = self._receive_data_size()
        #print('string size is', size)
        msg = self._receive_bytes(size)
        return msg

    def _receive_message_np(self):
        shape_len = self._receive_data_size()
        shape = [self._receive_data_size() for _ in range(shape_len)]
        size = np.prod(shape) * 4
        msg = self._receive_bytes(size)
        array = np.frombuffer(msg, dtype=np.float32).reshape(shape)
        return array

    def _receive_bytes(self, msg_size):
        msg = bytes() 
        while msg_size > 0:
            new_part = self.socket.recv(msg_size)
            msg_size -= len(new_part)
            msg += new_part
        return msg

    def _send_string(self, s):
        self.socket.sendall(struct.pack("I", len(s)))
        self.socket.sendall(s)

    def _default_hparams(self):
        default_dict = {
            'host': '127.0.0.1',
            'port': 65433,
        }
        parent_params = super(GCRemotePolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t=None, i_tr=None, images=None, state=None, goal_image=None):

        # Send current state
       	keys = ["images", "state", "t", "goal_image"]

        # Index into camera number 0 since we will assume we have one camera
        images = images[:, 0]
        goal_image = goal_image[0, 0]

	self._send_string('SOM')
	for key in keys:
	    if key == "images":
		message = images
	    elif key == "states":
		message = state
	    elif key == "t":
		message = np.array([t])
	    elif key == "goal_image":
		message = goal_image
            
            # Communication expects float32
            message = message.astype(np.float32)
	    message_string = message.tobytes()
	    self._send_string(key)
	    self.socket.sendall(struct.pack("I", len(message.shape)))
	    [self.socket.sendall(struct.pack("I", d)) for d in message.shape]
	    self.socket.sendall(message_string)
	self._send_string('EOM')

        # Receive action computed by remote policy
	action = self._receive_message_np()
        print('taking action', action)
        #action = action[:4]
        #action = np.zeros(4)
	return {'actions': action}
 

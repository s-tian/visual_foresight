""" This file defines an interface for a remote policy, which uses a remote procedure call (RPC) to outsource expensive computation """

import numpy as np
import rpyc

from visual_mpc.policy.policy import Policy


class RemotePolicy(Policy):

    def __init__(self, agentparams, policyparams, gpu_id, npgu):
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)
        self.agentparams = agentparams
        self.adim = agentparams['adim']

        # Start RPC connection
        try:
            print('Trying to connect to {} at port {}'.format(self._hp.address, self._hp.port))
            self.connection = rpyc.connect(self._hp.address, self._hp.port, config={
                "allow_all_attrs": True,
                'allow_pickle': True,
                'sync_request_timeout': 300,
            })
        except Exception as e:
            print('Failed to connect! Please make sure the policy server is running!')
            raise e

        self._remote_act = self.connection.root.act
        self.reset = self.connection.root.reset
        self.set_log_dir = self.connection.root.set_log_dir

    def _default_hparams(self):
        default_dict = {
            'address': 'none',
            'port': 0,
            'policy_type': RemotePolicy,
        }

        parent_params = super(RemotePolicy, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, *args, **kwargs):
        ret = self._remote_act(*args, **kwargs)
        for key in ret:
            if isinstance(ret[key], np.ndarray):
                ret[key] = np.array(ret[key])
        return ret


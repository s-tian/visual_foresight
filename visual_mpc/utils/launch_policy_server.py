import importlib.machinery
import importlib.util
import imp
import argparse
import rpyc
from rpyc.utils.server import ThreadedServer


class PolicyService(rpyc.Service):

    def __init__(self, policy):
        super(rpyc.Service, self).__init__()
        self.policy = policy

    def on_connect(self, conn):
        print('Policy service established connection with client...')
        print(conn)

    def on_disconnect(self, conn):
        print('Policy service disconnected with client...')
        print(conn)

    def exposed_set_log_dir(self, *args, **kwargs):
        return self.policy.set_log_dir(*args, **kwargs)

    def exposed_reset(self, *args, **kwargs):
        return self.policy.reset(*args, **kwargs)

    def exposed_act(self, *args, **kwargs):
        return self.policy.act(*args, **kwargs)


def load_config_from_file(filepath):
    hyperparams = imp.load_source('hyperparams', filepath)
    conf = hyperparams.config
    return conf

    hyperparams_file = filepath
    loader = importlib.machinery.SourceFileLoader('mod_hyper', hyperparams_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod.config


def launch_policy_server(cmd_args):
    port_num = cmd_args.port
    hp = load_config_from_file(cmd_args.experiment_hparams)
    print('Creating agent...')
    agent = hp['agent']['type'](hp['agent'])
    print('Creating policy object...')

    policy = hp['policy']['type'](agent._hp.values(), hp['policy'], cmd_args.gpu_id, cmd_args.ngpu)
    print('Launching server on port {}...'.format(port_num))
    server = ThreadedServer(PolicyService(policy), port=port_num, protocol_config={
        'allow_all_attrs': True,
        'allow_pickle': True,
        'sync_request_timeout': 300,
    })
    server.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launches a policy server based on the given arguments")
    parser.add_argument('experiment_hparams', type=str, help="file containing experiment hparams")
    parser.add_argument('--port', type=int, help="port number", default=18812)
    parser.add_argument('--gpu_id', type=int, help="gpu id", default=0)
    parser.add_argument('--ngpu', type=int, help="number of gpus to use", default=1)
    args = parser.parse_args()
    launch_policy_server(args)

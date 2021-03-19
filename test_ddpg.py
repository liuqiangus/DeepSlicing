import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup.utils.logx import EpochLogger
from spinup.utils.logx import restore_tf_graph
import numpy as np
import matplotlib.pyplot as matplt


def load_policy(fpath, itr='last', deterministic=False):

    tf.reset_default_graph()
    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]



    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action, sess


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    ep_action, ep_reward, ep_utility = [], [], []
    #logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    penalty = o[-env.num_res:]
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        ep_action.append(np.reshape(a, [env.num_res, env.UENum]))
        ep_reward.append(r)
        ep_utility.append(info)
        if d or (ep_len == max_ep_len):
            #logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    #logger.log_tabular('EpRet', with_min_and_max=True)
    #logger.log_tabular('EpLen', average_only=True)
    #logger.dump_tabular()

    #matplt.subplot(3,1,1)
    #matplt.plot(np.sum(ep_action, axis=1))
    #matplt.plot((penalty/env.Rmax)*np.ones(len(ep_reward)))
    #matplt.subplot(3,1,2)
    #matplt.plot(ep_reward)
    #matplt.subplot(3,1,3)
    #matplt.plot(ep_utility)
    # matplt.plot(np.sum(ep_action,axis=1))
    #print(np.sum(np.mean(ep_action, axis=0)), np.sum(penalty)/100)
    #matplt.show()

    return np.array(ep_reward), np.array(ep_action), np.array(ep_utility)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='1slice1')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic)
    #env = ResourceEnv()
    run_policy(env, get_action, args.len, args.episodes, args.norender)

    sess.close()

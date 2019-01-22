import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, microbatch_size=None):
        self.sess = sess = get_session()

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        self.OLDNEGLOGSTD = OLDNEGLOGSTD = tf.placeholder(tf.float32, [None, ac_space.shape[0]])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        self.H = H = tf.placeholder(tf.float32, [None])
        self.HADV = HADV = tf.placeholder(tf.float32, [None])

        neglogpac = train_model.pd.neglogp(A)
        neglogstd = -train_model.pd.logstd
        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        ### H loss
        hpred = train_model.h
        h_loss = tf.square(hpred - H)

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        v_ratio = tf.reduce_mean(OLDNEGLOGSTD - neglogstd, 1)
        v_ratio_t1 = tf.concat([v_ratio[1:], v_ratio[-1:]],axis=0)
        # Defining Loss = - J is equivalent to max J
        pg_losses = -(ADV + ent_coef * HADV + LR * v_ratio_t1) * ratio
        CLIPRANGE_ALPHA = 0.0003/(LR + 1e-6)
        pg_losses2 = -(ADV + ent_coef * HADV + LR * tf.clip_by_value(v_ratio_t1, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)) * \
                     tf.clip_by_value(ratio * v_ratio, 1.0 - CLIPRANGE / 1.5, 1.0 + CLIPRANGE / 1.5) / \
                     tf.clip_by_value(v_ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        ########### 2 -> 0.2   2.1 -> 0.5
        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss + h_loss + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        if MPI is not None:
            self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac','LR']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac, LR]


        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, hreturns, masks, actions, values, hvalues, neglogpacs, neglogstd, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        ### here we calculate del_hadvs = E(sigma) + yH(s') - H(s)
        ### A = R + yV(s') - V(s) + h_coef * (E(sigma) + yH(s') - H(s))
        ### returns = A + V
        ### ---------RUNNER above-----------
        ### returns = A + V - V    for V network
        ### advs = A + V + H  used for pi_mean network

        advs = returns - values
        hadvs = hreturns - hvalues
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        hadvs = (hadvs - hadvs.mean()) / (hadvs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDNEGLOGSTD : neglogstd,
            self.OLDVPRED : values,
            self.H: hreturns,
            self.HADV: hadvs,
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]


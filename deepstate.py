import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np
import argparse
import sys
from collections import namedtuple


parameter_class = namedtuple('parameters', ['A', 'C', 'Q', 'R', 'mu', 'sigma'])


class DeepState(tf.keras.Model):
    """
    This class defines a Kalman Filter (Linear Gaussian State Space model) 
    parameterized by a RNN.
    """

    def __init__(self,
                 dim_z, 
                 seq_len,
                 dim_y=1,
                 dim_u=0, 
                 rnn_units=32,
                 no_use_cudnn_rnn=True,
                 **kwargs):
        super(DeepState, self).__init__()

        self.seq_len = seq_len
        self.dim_z = dim_z
        self.dim_y = dim_y

        # Create model
        if no_use_cudnn_rnn:
            self.rnn = layers.LSTM(rnn_units, 
                                   return_sequences=True)
        else:
            self.rnn = layers.CuDNNLSTM(rnn_units, 
                                        return_sequences=True)

        self.A = layers.Dense(dim_z*dim_z)
        self.C = layers.Dense(dim_z)
        self.Q = layers.Dense(dim_z * dim_z)
        self.R = layers.Dense(dim_y * dim_y)
        self.mu = layers.Dense(dim_z)
        self.sigma = layers.Dense(dim_z * dim_z)

        self._alpha_sq = tf.constant(1., dtype=tf.float32)  # fading memory control
        self.M = 0                                          # process-measurement cross correlation

        # identity matrix
        self._I = tf.eye(dim_z, name='I')

        self.state = kwargs.pop('state', None)
        self.log_likelihood = None

    def call(self, x, y):
        # Create mask of ones as we don't use it right now
        self.mask = tf.ones((y.shape[0], 1))

        # Compute RNN outputs
        output = self.rnn(x)

        # Get initial state
        mu = tf.reshape(self.mu(output[:, 1]), (-1, self.dim_z))
        sigma = tf.reshape(self.sigma(output[:, 1]), (-1, self.dim_z, self.dim_z))

        # Get parameters for the sequence
        output = tf.reshape(output, (-1, output.shape[2]))
        A = tf.reshape(self.A(output), (-1, self.seq_len, self.dim_z, self.dim_z), 'A')
        C = tf.reshape(self.C(output), (-1, self.seq_len, self.dim_y, self.dim_z), 'C')
        Q = tf.reshape(self.Q(output), (-1, self.seq_len, self.dim_z, self.dim_z), 'Q')
        R = tf.reshape(self.R(output), (-1, self.seq_len, self.dim_y, self.dim_y), 'R')

        # self.parameters = list((A, C, Q, R, mu, sigma))
        self.parameters = parameter_class(A, C, Q, R, mu, sigma)
        forward_states = self.compute_forwards(y, self.parameters)
        backward_states = self.compute_backwards(forward_states, self.parameters)

        return backward_states

    def forward_step_fn(self, params, y, A, C, Q, R):
        """
        Forward step over a batch
        """
        mu_pred, Sigma_pred, mu_t, Sigma_t = params

        # Residual
        y_pred = tf.squeeze(tf.matmul(C, tf.expand_dims(mu_pred, 2)))       # (bs, dim_y)
        r = tf.reshape(y - y_pred, (-1, 1), name='residual')                # (bs, dim_y)

        # project system uncertainty into measurement space
        S = tf.matmul(tf.matmul(C, Sigma_pred), C, transpose_b=True) + R    # (bs, dim_y, dim_y)

        S_inv = tf.matrix_inverse(S)
        K = tf.matmul(tf.matmul(Sigma_pred, C, transpose_b=True), S_inv)    # (bs, dim_z, dim_y)

        # For missing values, set to 0 the Kalman gain matrix
        K = tf.multiply(tf.expand_dims(self.mask, 2), K)

        # Get current mu and Sigma
        mu_t = mu_pred + tf.squeeze(tf.matmul(K, tf.expand_dims(r, 2)))     # (bs, dim_z)
        I_KC = self._I - tf.matmul(K, C)                                    # (bs, dim_z, dim_z)
        Sigma_t = tf.matmul(tf.matmul(I_KC, Sigma_pred), I_KC, transpose_b=True)  # (bs, dim_z, dim_z)
        Sigma_t += K * R * tf.transpose(K, [0, 2, 1])

        # Prediction
        mu_pred = tf.squeeze(tf.matmul(A, tf.expand_dims(mu_t, 2))) 
        # mu_pred = mu_pred + tf.squeeze(tf.matmul(B, tf.expand_dims(u, 2)))
        Sigma_pred = tf.scalar_mul(self._alpha_sq, tf.matmul(tf.matmul(A, Sigma_t), A, transpose_b=True) + Q)

        return mu_pred, Sigma_pred, mu_t, Sigma_t

    def backward_step_fn(self, params, inputs):
        """
        Backwards step over a batch, to be used in tf.scan
        :param params:
        :param inputs: (batch_size, variable dimensions)
        :return:
        """
        mu_back, Sigma_back = params
        mu_pred_tp1, Sigma_pred_tp1, mu_filt_t, Sigma_filt_t, A = inputs

        J_t = tf.matmul(tf.transpose(A, [0, 2, 1]), tf.matrix_inverse(Sigma_pred_tp1))
        J_t = tf.matmul(Sigma_filt_t, J_t)

        mu_back = mu_filt_t + tf.matmul(J_t, mu_back - mu_pred_tp1)
        Sigma_back = Sigma_filt_t + tf.matmul(J_t, tf.matmul(Sigma_back - Sigma_pred_tp1, J_t, adjoint_b=True))

        return mu_back, Sigma_back

    def compute_forwards(self, y, parameters):
        # Set initial state
        sigma = parameters.sigma
        mu = parameters.mu
        params = [mu, sigma, mu, sigma]

        # Step through the sequence
        states = list()
        for i in range(self.seq_len):
            params = self.forward_step_fn(params,
                                          y[:, i],
                                          parameters.A[:, i],
                                          parameters.C[:, i],
                                          parameters.Q[:, i],
                                          parameters.R[:, i])
            states.append(params)

        # Restructure to tensors of shape=(seq_len, batch_size, dim_z)
        states = list(map(list, zip(*states)))
        states = [tf.stack(state, axis=0) for state in states]
        return states

    def compute_backwards(self, forward_states, parameters):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt = forward_states
        mu_pred = tf.expand_dims(mu_pred, 3)
        mu_filt = tf.expand_dims(mu_filt, 3)
        # The tf.scan below that does the smoothing is initialized with the filtering distribution at time T.
        # following the derivation in Murphy's book, we then need to discard the last time step of the predictive
        # (that will then have t=2,..T) and filtering distribution (t=1:T-1)
        states_scan = [mu_pred[:-1],
                       Sigma_pred[:-1],
                       mu_filt[:-1],
                       Sigma_filt[:-1],
                       tf.transpose(parameters.A, (1, 0, 2, 3))[:-1]]

        # Reverse time dimension
        dims = [0]
        for i, state in enumerate(states_scan):
            states_scan[i] = tf.reverse(state, dims)

        # Transpose list of lists
        states_scan = list(map(list, zip(*states_scan)))

        # Init params
        params = [mu_filt[-1], Sigma_filt[-1]]

        backward_states = list()
        for i in range(self.seq_len - 1):
            params = self.backward_step_fn(params,
                                           states_scan[i])
            backward_states.append(params)

        # Restructure to tensors of shape=(seq_len, batch_size, dim_z)
        backward_states = list(map(list, zip(*backward_states)))
        backward_states = [tf.stack(state, axis=0) for state in backward_states]

        # Reverse time dimension
        backward_states = list(backward_states)
        dims = [0]
        for i, state in enumerate(backward_states):
            backward_states[i] = tf.reverse(state, dims)

        # Add the final state from the filtering distribution
        backward_states[0] = tf.concat([backward_states[0], mu_filt[-1:, :, :, :]], axis=0)
        backward_states[1] = tf.concat([backward_states[1], Sigma_filt[-1:, :, :, :]], axis=0)

        # Remove extra dimension in the mean
        backward_states[0] = backward_states[0][:, :, :, 0]

        return backward_states

    def get_elbo(self, states, y, mask):
        A, C, Q, R, mu, sigma = self.parameters
        mu_smooth = states[0]
        Sigma_smooth = states[1]

        # Sample from smoothing distribution
        jitter = 1e-2 * tf.eye(Sigma_smooth.shape[-1], batch_shape=tf.shape(Sigma_smooth)[0:-2])
        # mvn_smooth = tf.contrib.distributions.MultivariateNormalTriL(mu_smooth, Sigma_smooth + jitter)
        mvn_smooth = tfp.distributions.MultivariateNormalTriL(mu_smooth, tf.cholesky(Sigma_smooth + jitter))
        z_smooth = mvn_smooth.sample()

        ## Transition distribution \prod_{t=2}^T p(z_t|z_{t-1}, u_{t})
        # We need to evaluate N(z_t; Az_tm1 + Bu_t, Q), where Q is the same for all the elements
        # z_tm1 = tf.reshape(z_smooth[:, :-1, :], [-1, self.dim_z])
        # Az_tm1 = tf.transpose(tf.matmul(self.A, tf.transpose(z_tm1)))
        Az_tm1 = tf.reshape(tf.matmul(A[:, :-1], tf.expand_dims(z_smooth[:, :-1], 3)), [-1, self.dim_z])

        # Remove the first input as our prior over z_1 does not depend on it
        # u_t_resh = tf.reshape(u, [-1, self.dim_u])
        # Bu_t = tf.transpose(tf.matmul(self.B, tf.transpose(u_t_resh)))
        # Bu_t = tf.reshape(tf.matmul(B[:, :-1], tf.expand_dims(u[:, 1:], 3)), [-1, self.dim_z])
        mu_transition = Az_tm1 # + Bu_t
        z_t_transition = tf.reshape(z_smooth[:, 1:, :], [-1, self.dim_z])

        # MultivariateNormalTriL supports broadcasting only for the inputs, not for the covariance
        # To exploit this we then write N(z_t; Az_tm1 + Bu_t, Q) as N(z_t - Az_tm1 - Bu_t; 0, Q)
        trans_centered = z_t_transition - mu_transition
        mvn_transition = tfp.distributions.MultivariateNormalTriL(tf.zeros(self.dim_z), tf.cholesky(Q))
        log_prob_transition = mvn_transition.log_prob(trans_centered)

        ## Emission distribution \prod_{t=1}^T p(y_t|z_t)
        # We need to evaluate N(y_t; Cz_t, R). We write it as N(y_t - Cz_t; 0, R)
        # z_t_emission = tf.reshape(z_smooth, [-1, self.dim_z])
        # Cz_t = tf.transpose(tf.matmul(self.C, tf.transpose(z_t_emission)))
        Cz_t = tf.reshape(tf.matmul(C, tf.expand_dims(z_smooth, 3)), [-1, self.dim_y])

        y_t_resh = tf.reshape(y, [-1, self.dim_y])
        emiss_centered = y_t_resh - Cz_t
        mvn_emission = tfp.distributions.MultivariateNormalTriL(tf.zeros(self.dim_y), tf.cholesky(R))
        mask_flat = tf.reshape(mask, (-1, ))
        log_prob_emission = mvn_emission.log_prob(emiss_centered)
        log_prob_emission = tf.multiply(mask_flat, log_prob_emission)

        ## Distribution of the initial state p(z_1|z_0)
        z_0 = z_smooth[:, 0, :]
        mvn_0 = tfp.distributions.MultivariateNormalTriL(mu, tf.cholesky(sigma))
        log_prob_0 = mvn_0.log_prob(z_0)

        # Entropy log(\prod_{t=1}^T p(z_t|y_{1:T}, u_{1:T}))
        entropy = - mvn_smooth.log_prob(z_smooth)
        entropy = tf.reshape(entropy, [-1])
        # entropy = tf.zeros(())

        # Compute terms of the lower bound
        # We compute the log-likelihood *per frame*
        num_el = tf.reduce_sum(mask_flat)
        log_probs = [tf.truediv(tf.reduce_sum(log_prob_transition), num_el),
                     tf.truediv(tf.reduce_sum(log_prob_emission), num_el),
                     tf.truediv(tf.reduce_sum(log_prob_0), num_el),
                     tf.truediv(tf.reduce_sum(entropy), num_el)]

        kf_elbo = tf.reduce_sum(log_probs)

        return kf_elbo, log_probs, z_smooth


def generate_data(samples, seq_len):
    y = tf.random.normal((samples, seq_len)) + tf.linspace(0., 1., seq_len)
    x = tf.random.normal((samples, seq_len, 1))
    x = tf.concat((x, tf.reshape(y, (samples, seq_len, 1))*2), axis=2)
    return x, y


def loss_fn(model, inputs, targets, mask):
    states = model(inputs, targets)
    kf_elbo, log_probs, z_smooth = model.get_elbo(states, targets, mask)
    return -kf_elbo


def train(model, optimizer, train_data, train_target, mask):
    def model_loss(inputs, targets):
        return loss_fn(model, inputs, targets, mask)

    grad_fn = tfe.implicit_gradients(model_loss)
    grads_and_vars = grad_fn(train_data, train_target)
    optimizer.apply_gradients(grads_and_vars)


def evaluate(model, data, targets, mask):
    """evaluate an epoch."""

    loss = loss_fn(model, data, targets, mask)
    return loss


def main(_):
    tf.enable_eager_execution()

    model = DeepState(dim_z=4, seq_len=FLAGS.seq_len)

    mask = tf.ones((100, 1))
    train_data, train_target = generate_data(100, FLAGS.seq_len)
    test_data, test_target = generate_data(100, FLAGS.seq_len)
    learning_rate = tf.Variable(0.005, name="learning_rate")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    for _ in range(FLAGS.epoch):
        train(model, optimizer, train_data, train_target, mask)
        loss = evaluate(model, test_data, test_target, mask)
        print(f'Test loss: {loss}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data-path",
      type=str,
      default="")
  parser.add_argument(
      "--logdir", type=str, default="", help="Directory for checkpoint.")
  parser.add_argument("--epoch", type=int, default=20, help="Number of epochs.")
  parser.add_argument("--batch-size", type=int, default=20, help="Batch size.")
  parser.add_argument(
      "--seq-len", type=int, default=35, help="Sequence length.")
  parser.add_argument(
      "--hidden-dim", type=int, default=200, help="Hidden layer dimension.")
  parser.add_argument(
      "--num-layers", type=int, default=2, help="Number of RNN layers.")
  parser.add_argument(
      "--dropout", type=float, default=0.2, help="Drop out ratio.")
  parser.add_argument(
      "--clip", type=float, default=0.25, help="Gradient clipping ratio.")
  parser.add_argument(
      "--no-use-cudnn-rnn",
      action="store_true",
      default=True,
      help="Disable the fast CuDNN RNN (when no gpu)")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
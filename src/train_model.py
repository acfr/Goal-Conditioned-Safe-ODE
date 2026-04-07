'''
Train the diffeomorphism as Equation (35) in the paper for 2D corridor.

Author: Dechuan Liu (April 2026)
'''

import optax
import scipy.io
import os
from sklearn.model_selection import ParameterGrid
import jax.random as random
import jax.numpy as jnp
import orbax.checkpoint
import jax
from flax.training import train_state, orbax_utils

# Add the test folder to sys.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils import DataLoader_maze, normalize_data, map_obstacle
from robustnn.plnet_jax import PLNet
from robustnn.bilipnet_jax import BiLipNet

def get_fitness_loss(model, 
                     optimal_point = None,
                     is_optimal=False,
                     is_evaluation = False, # define if it is evaluation
                     ):
    # regression
    @jax.jit
    def fitloss(state, params, x, y):
        # Apply the model to get predictions
        if is_optimal:
            yh = model.apply(params, x, optimal_point)
        else:
            yh = model.apply(params, x)
        
        # take gradient value
        loss = jnp.square(jax.nn.relu(jnp.where(y[:, 0] == 0, 
                                                yh - y[:, 1],  
                                                y[:, 1] - yh))).mean()

        if is_evaluation:
            return [loss]
        return loss

    return fitloss

def train_with_flexible_loss(
	rng,
	model,
	data,
	fitness_func,
	fitness_eval_func,
	name: str = 'bilipnet',
	train_dir: str = './results/rosenbrock-nd',
	lr_max: float = 1e-3,
	epochs: int = 600,
	is_batched_input: bool= False,
):

	ckpt_dir = f'{train_dir}/ckpt'
	os.makedirs(ckpt_dir, exist_ok=True)

	data_dim = data['data_dim']
	train_batches = data['train_batches']
	train_batch_size = data['train_batch_size']

	idx_shp = (train_batches, train_batch_size)
	train_size = train_batches * train_batch_size

	rng, rng_model = random.split(rng)
	
	# special batch considered for J and R
	if is_batched_input:
		params = model.init(rng_model, jnp.ones((1, data_dim)))
	else:
		params = model.init(rng_model, jnp.ones(data_dim))
	
	param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
	print(f'model: {name}, size: {param_count/1000000:.2f}M')

	total_steps = train_batches * epochs
	scheduler = optax.linear_onecycle_schedule(transition_steps=total_steps, 
										peak_value=lr_max,
										pct_start=0.25, 
										pct_final=0.7,
										div_factor=10., 
										final_div_factor=200.)
	opt = optax.adam(learning_rate=scheduler)
	model_state = train_state.TrainState.create(apply_fn=model.apply,
												params=params,
												tx=opt)
	
	@jax.jit
	def train_step(state, x, y):
		grad_fn = jax.value_and_grad(fitness_func, argnums=1)
		loss, grads = grad_fn(state, state.params, x, y)
		state = state.apply_gradients(grads=grads)
		return state, loss 
	
	train_loss, val_loss = [], []
	Lipmin, Lipmax, Tau = [], [], []
	for epoch in range(epochs):
		rng, rng_idx = random.split(rng)
		idx = random.permutation(rng_idx, train_size)
		idx = jnp.reshape(idx, idx_shp)
		tloss = 0. 
		for b in range(train_batches):
			x = data['xtrain'][idx[b, :], :] 
			y = data['ytrain'][idx[b, :]]
			model_state, loss = train_step(model_state, x, y)
			tloss += loss
			
		tloss /= train_batches
		train_loss.append(tloss)

		vloss = fitness_eval_func(model_state, model_state.params, data['xtest'], data['ytest'])
		val_loss.append(vloss)

		lipmin, lipmax, tau = model.get_bounds(model_state.params)
		Lipmin.append(lipmin)
		Lipmax.append(lipmax)
		Tau.append(tau)

		print(f'Epoch: {epoch+1:3d} | loss: {tloss:.3f}/{vloss}, tau: {tau:.3f}, Lip: {lipmin:.3f}/{lipmax:.2f}')
		
	eloss = fitness_eval_func(model_state, model_state.params, data['xeval'], data['yeval'])
	print(f'{name}: eval loss: {eloss}')

	data['train_loss'] = jnp.array(train_loss)
	data['val_loss'] = jnp.array(val_loss)
	data['lipmin'] = jnp.array(Lipmin)
	data['lipmax'] = jnp.array(Lipmax)
	data['tau'] = jnp.array(Tau)
	data['eval_loss'] = eloss

	scipy.io.savemat(f'{train_dir}/data.mat', data)

	orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
	save_args = orbax_utils.save_args_from_target(model_state.params)
	orbax_checkpointer.save(f'{ckpt_dir}/params', model_state.params, save_args=save_args)


def training(lr_max, rng, root_dir, params, file_path, x_range, y_range):
    data_dim = params['data_dim']
    zero_point = params['zero_point']
    layer_size = params['layer_size']
    depth = params['depth']
    mu = params['mu']
    nu = params['nu']
    epoch = params['epoch']
    inf_replace = params['inf_replace']
    training_samples = params['training_samples']
    train_batch_size = params['train_batch_size']
    train_batches = int(training_samples/train_batch_size) + 1
    normalized_range = params['normalized_range']

    # load data
    data_loader = DataLoader_maze(file_path, 
                                  inf_replaced=inf_replace)
    data = data_loader.get_data(train_batch_size=train_batch_size,
                                train_batches=train_batches)
    
    # normalize data
    rrt_range = data['y_range']

    y = data['y']
    y[:,1] = normalize_data(data['y'][:,1], rrt_range, normalized_range)

    y_train = data['ytrain']
    y_train[:,1] = normalize_data(data['ytrain'][:,1], rrt_range, normalized_range)

    y_test = data['ytest']
    y_test[:,1] = normalize_data(data['ytest'][:,1], rrt_range, normalized_range)

    y_eval = data['yeval']
    y_eval[:,1] = normalize_data(data['yeval'][:,1], rrt_range, normalized_range)

    data = {
            "x": normalize_data(data['x'], x_range, normalized_range),
            "y": y,
            "xtrain": normalize_data(data['xtrain'], x_range, normalized_range), 
            "ytrain": y_train, 
            "xtest": normalize_data(data['xtest'], x_range, normalized_range), 
            "ytest": y_test, 
            "xeval": normalize_data(data['xeval'], x_range, normalized_range),
            "yeval": y_eval,
            "data_size": data['data_size'],
            "train_batches": data['train_batches'],
            "train_batch_size": data['train_batch_size'],
            "test_batches": data['test_batches'],
            "test_batch_size": data['test_batch_size'],
            "eval_batches": data['eval_batches'],
            "eval_batch_size": data['eval_batch_size'],
            "data_dim": data['data_dim']
        }
    
    # create models
    block = BiLipNet(input_size=data_dim, 
                     units=layer_size, 
                     depth=depth, 
                     mu=mu, 
                     nu=nu)
          
    model_pl = PLNet(BiLipBlock=block, optimal_point=jnp.array(zero_point))

    # set fitness function
    fitness_func_pl = get_fitness_loss(model_pl,
            optimal_point=jax.numpy.array(zero_point),
            is_optimal=True,
            is_evaluation=False)
    
    fitness_eval_func_pl = get_fitness_loss(model_pl,
            optimal_point=jax.numpy.array(zero_point),
            is_optimal=True,
            is_evaluation=True)


    # names
    prefix = f'model'
    train_dir_pl = f'{root_dir}/{prefix}/pl'

    train_with_flexible_loss(rng, model_pl, data, 
                                fitness_func=fitness_func_pl,
                                fitness_eval_func=fitness_eval_func_pl,
                                name=name, 
                                train_dir=train_dir_pl, 
                                lr_max=lr_max, 
                                epochs=epoch)

if __name__ == "__main__":
    file_path = os.getcwd()+'/src/data/2D-corridor/map_10000_barrier_value.mat'

    x_range = [-8, 8]
    y_range = [-8, 8]

    # given constant
    lr_max = 1e-4
    rng = random.PRNGKey(42)
    name = '2D-corridor'
    root_dir = os.getcwd() + f'/results/{name}'
    

    params = {
        'data_dim': 2,
        'mu': 0.1,
        'nu': 128,
        'depth': 12,
        'epoch': 1500,
        'layer_size': [128]*4,
        'inf_replace': 6000,
        'training_samples': 5000,
        'train_batch_size': 16,
        'inver_and_update_num': 500,
        'zero_point':(0.5,7/8),
        'normalized_range':[0,1],
    }

    training(lr_max, rng, root_dir, params, 
                file_path, x_range=x_range, y_range=y_range)
        



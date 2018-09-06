import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, models, datasets
from torch.nn.parameter import Parameter 
import numpy as np
import argparse

#height and width of image that can be passed through model
#to be attcked
#default = imagenet
ORG_HEIGHT = 224
ORG_WIDTH = 224
#height and width of the data of adversarial task
ADV_HEIGHT = 28
ADV_WIDTH = 28


def parse_arguments():
	#parse arguments
	args = argparse.ArgumentParser()
	args.add_argument("--model", help="pretrained model to be reprogrammed (i.e resnet, inception")
	args.add_argument("--mode", help="[train, validate, inference]")
	args.add_argument("--data", help="data for adversarial task i.e mnist")
	args.add_argument("--cuda", action='store_true', help="train on gpu")
	args.add_argument("--retrain", action='store_true', help="retrain using some previous weights")
	args.add_argument("--batch_size", type=int, help="batch size for training")
	args.add_argument("--epochs", type=int, help="number of epochs to train")
	args.add_argument("--pretrained_model", help="path to a pretrained model")
	args.add_argument("--input_img", help="path to input image for inference")
	args = args.parse_args()
	return args


def load_adversarial_task_data(data, batch_size, validate=False):

	""" Load data for adversarial task (i.e counting squares, MNIST
		This script supports MNIST only, but you can write your
		dataloader to laod custom data.
		
		write your dataloader as shown here-
		https://github.com/utkuozbulak/pytorch-custom-dataset-examples 
		then use it as shown below in comments
	"""
	if data == "mnist":

		data_loader = torch.utils.data.DataLoader(
			datasets.MNIST('../data', train=(not validate), download=True,
				transform=transforms.Compose([
					transforms.ToTensor()])),
			batch_size=batch_size, shuffle=True)

	elif data == "custom":
		pass
		"""if your custom dataloader's name is CustomDataset then
		if validate:
			data_loader = CustomDataset(arguments)
		else:
			data_loader = CustomDataset(arguments) 
		"""
	return data_loader


def load_model_attack( model):
	""" Load model to be attcked.
		if using dataset other than imagenet
		then don't forget to change ORG_WIDTH and ORG_HEIGHT
		after loading the pretrained model"""

	if model == "resnet50":
		print("Loading pre-trained resnet50 model")
		model = models.resnet50(pretrained=True)

	elif model == "inceptionv3":
		print("Loading pre-trained inceptionv3 model")
		model = models.inceptionv3(pretrained=True)

	else:
		print("Error: invalid model to be attacked: {}".format(args.model))

	for param in model.parameters():
		param.requires_grad = False
	return model


class AdversarialProgram(nn.Module):


	def __init__(self, model, batch_size):

		super(AdversarialProgram, self).__init__()

		self.model = load_model_attack(model)
		self.W = Parameter(torch.randn(3, ORG_HEIGHT, ORG_WIDTH), requires_grad=True)
		self.mask()
		self.batchSize = batch_size
		self.set_mean_std()

	def H_g(self, label, dataset="imagenet"):
		""" Implement function Hg (as given in the paper)
			: a hard coded mapping that returns label with
			the same shape as (i.e) mnist label"""

		if dataset == "imagenet":
			#assign first ten imagenet labels
			return label[:,:10]

	def forward(self, adv_data):
		adv_data = adv_data.repeat(1,3,1,1)
		# data = torch.zeros(self.batchSize, 3, ORG_HEIGHT, ORG_WIDTH)
		X = adv_data.data.new(self.batchSize, 3, ORG_HEIGHT, ORG_WIDTH)
		X[:] = 0
		X[:, :, self.h_lower:self.h_upper, self.w_lower:self.w_upper ] = adv_data.data.clone()
		# tanh = nn.Tanh()
		P = torch.sigmoid(self.W * self.M)
		# if self.if_cuda:
		# 	P.type('torch.cuda.FloatTensor')
		X_adv = X + P
		X_adv_norm = (X_adv - self.mean) / self.std
		out = self.model(X_adv_norm)
		out = self.H_g(F.softmax(out, dim=1), dataset="imagenet")
		return out

	def mask(self):
		m = torch.ones(3, ORG_HEIGHT, ORG_WIDTH)
		x_center, y_center = ORG_WIDTH//2, ORG_HEIGHT//2
		self.h_lower = y_center - (ADV_HEIGHT//2)
		self.h_upper = y_center + (ADV_HEIGHT//2)
		self.w_lower = x_center - (ADV_WIDTH//2)
		self.w_upper = x_center + (ADV_WIDTH//2)
		m[:,self.h_lower:self.h_upper, self.w_lower:self.w_upper] = 0
		self.M = Parameter(m, requires_grad=False)       

	def set_mean_std(self):
		mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
		std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
		self.mean = torch.from_numpy(mean)
		self.std = torch.from_numpy(std)


def get_custom_loss(bce_loss, output, target, w):
	return bce_loss + 0.05 * (torch.norm(w) ** 2)

def get_w(model):
	for param in model.parameters():
		if param.requires_grad: #only w requires grad
 			return param

def generate_target(batch_size, target):
  t = torch.zeros(batch_size, 10)
  for i,n in enumerate(target):
    t[i][n] = 1
  return t

# def save_model_fn(epoch, model, optimizer, name):
# 	#save the model
# 	state = {
# 	    'epoch': epoch+1,
# 	    'state_dict': model.state_dict(),
# 	    'optimizer': optimizer.state_dict(),
# 	}

# 	torch.save(state, name)
# 	print("Model- {} saved successfully".format(name))


def train(model, batch_size, if_cuda, data, epochs, retrain, pretrained_model):

	#load model
	adv_program = AdversarialProgram(model, batch_size)
	if if_cuda:
		adv_program.cuda()
		adv_program.mean.type('torch.cuda.FloatTensor')
		adv_program.std.type('torch.cuda.FloatTensor')

	#load orginal data
	data_loader = load_adversarial_task_data(data, batch_size, validate=False)

	#loss and optimizer
	loss = nn.BCELoss()
	optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, adv_program.parameters()), lr=0.05)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.87)

	if retrain:
		state = torch.load(pretrained_model)
		adv_program.load_state_dict(state['state_dict'])

	for epoch in range(epochs+1):
		lr_scheduler.step()
		for idx, (data, target) in enumerate(data_loader):

			#wrap data into adversarial program
			data = Variable(data, requires_grad=False)
			target = generate_target(batch_size, target)
			target = Variable(target, requires_grad=False)

			if if_cuda:
				data, target = data.cuda(), target.cuda()


			#pass it through the model
			output = adv_program(data)

			#compute loss, backpropagate

			custom_loss = get_custom_loss(loss(output, target),output, target, get_w(adv_program))
			optimizer.zero_grad()
			custom_loss.backward()
			#optimize W
			optimizer.step()
			print("Iteration: {} Loss: {}".format(idx, custom_loss))

		print("Epoch {} completed. Saving model w_{}.pth.tar".format(epoch, epoch))
		name = "w_{}.pth.tar" % (epoch)
		save_model_fn(epoch, adv_program, optimizer, name)
		

def validate():

	#load original data
	data_loader = load_adversarial_task_data(args.data, validate=True)

	#wrap it into adv program

	#pass it through the model

	#calculate accuracy



def get_single_image(dataset):
	data_loader = load_adversarial_task_data(dataset, 1)
	i = int(np.random.randint(0,59999,1))
	data, target = data_loader.__getitem__(i)
	return data, target





def inference(model, pretrained_model, data):
	"""
	Accept pretrained "w" and a data sample as an input
	pass through the model and return output

	"""

	adv_program = AdversarialProgram(model, 1)
	adv_program.load_state_dict(torch.load("pretrained/w_9.pth.tar", map_location='cpu'), strict=False)
	input_img, true_label = get_single_image(data)
	output = adv_program(Variable(input_img, requires_grad=False))
	print("Prediction : {}    True Label : {} ".format(output, true_label))



if __name__ == '__main__':
	args = parse_arguments()
	if args.mode == "train":
		train(args.model, args.batch_size, args.cuda, args.data, args.epochs, args.retrain, args.pretrained_model)
	elif args.mode == "validate":
		pass
	elif args.mode == "inference":
		inference(args.model, args.pretrained_model, args.data)
	else:
		print("Error: Invalid mode {}".format(args.mode))
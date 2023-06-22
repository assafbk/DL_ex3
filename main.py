import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import os

from sklearn import svm
from joblib import dump, load

import pandas as pd

from VAE import VAE
import svm_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_vae(model, train_loader, optimizer, device, args, epoch):
    cur_results_dir = os.path.join(args.results_dir, 'epoch_{}'.format(epoch))
    if not os.path.exists(cur_results_dir):
        os.mkdir(cur_results_dir)
    model.train()
    total_loss = 0
    for i, (x, _) in enumerate(train_loader):
        x = x.squeeze()
        x = x.to(device)
        optimizer.zero_grad()
        x_hat, mu, sigma = model(x)
        loss, decoder_loss, encoder_loss = VAE.vae_loss(x, mu, sigma, x_hat)
        loss.backward()
        optimizer.step()
        total_loss += loss

        if i%50 == 0:
            print('train iter {}/{}: loss = {:.2f}, decoder loss = {:.2f}, encoder loss = {:.2f}'.format(i,len(train_loader), loss, decoder_loss, encoder_loss))
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(x[1,:,:].cpu())
            axarr[1].imshow(x_hat[1,:,:].detach().cpu())
            plt.savefig(os.path.join(cur_results_dir,'train_iter_{}.png'.format(i)))
            plt.close(f)

    return total_loss

def test_vae(model, test_loader, device, args, epoch):
    cur_results_dir = os.path.join(args.results_dir, 'epoch_{}'.format(epoch))
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.squeeze()
            x = x.to(device)
            x_hat, mu, sigma = model(x)
            loss, decoder_loss, encoder_loss = VAE.vae_loss(x_hat, mu, sigma, x)
            total_loss += loss
            if i % 30 == 0:
                print('test iter {}/{}: loss = {:.2f}, decoder loss = {:.2f}, encoder loss = {:.2f}'.format(i, len(test_loader), loss, decoder_loss, encoder_loss))
                f, axarr = plt.subplots(1, 2)
                axarr[0].imshow(x[1, :, :].cpu())
                axarr[1].imshow(x_hat[1, :, :].detach().cpu())
                plt.savefig(os.path.join(cur_results_dir,'test_iter_{}'.format(i)))
                plt.close(f)

    return total_loss

def train_svm(num_of_samples_per_class, num_of_classes, vae_model, train_loader, im_h, im_w):
    samples, labels = svm_utils.sample_balanced_data(num_of_samples_per_class, num_of_classes, train_loader, im_h, im_w)
    with torch.no_grad():
        samples_emb, _ = vae_model.encode(samples.to(device))
    svm_model = svm.SVC(kernel=args.svm_kernel)
    svm_model.fit(samples_emb.cpu(), labels)
    return svm_model

def test_svm(svm_model, vae_model, test_loader):
    total_correct = 0
    for x,y in test_loader:
        with torch.no_grad():
            x_emb,_ = vae_model.encode(x.to(device))
        y_hat = torch.Tensor(svm_model.predict(x_emb.cpu()))
        total_correct += torch.sum(y_hat == y)

    return total_correct/(len(test_loader)*test_loader.batch_size)

def parse_args():
    parser = argparse.ArgumentParser(
        description='''Classify FashionMNIST using Latent Feature Discriminate Model.\nUsage examples:
    train - 'python3 main.py --phase train --train_vae --train_svm --epochs 30'
    test the model - 'python3 main.py --phase test --vae_model_path <\path to vae model> --svm_models_dir <\path to dir of svm models>' ''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='./data', help='Directory for data')
    parser.add_argument('-o', '--output', type=str, default='./dump', help='Directory for output')
    parser.add_argument('-res', '--results_dir', type=str, default='./results', help='Directory for results')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train/test')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Training number of epochs')
    parser.add_argument('--vae_model_path', type=str, default=None, help='Path to saved vae model, test only')
    parser.add_argument('--svm_models_dir', type=str, default=None, help='Path to dir of saved svm models, test only')
    parser.add_argument('--show', action='store_true', default=False, help='Show plots')
    parser.add_argument('--train_vae', action='store_true', help='train vae model')
    parser.add_argument('--train_svm', action='store_true', help='train svm classifier')
    parser.add_argument('--svm_kernel', type=str, default='rbf', help='rbf\linear\poly')
    parser.add_argument('--dataset', type=str, default='FashionMNIST', help='MNIST\FashionMNIST')

    return parser.parse_args()

def get_svm_model(svm_models_dir, wanted_num_of_samples, dataset):
    all_models = os.listdir(svm_models_dir)
    for s in all_models:
        if s.startswith(str(wanted_num_of_samples)+'_samples_'+dataset):
            return load(os.path.join(svm_models_dir,s))

    raise(f'invalid svm model name {s} in {svm_models_dir}')

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    torch.manual_seed(123)
    batch_size = 64

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_epochs = args.epochs

    if args.phase == "train":
        if args.dataset == 'MNIST':
            train_dataset = torchvision.datasets.MNIST(root=args.input, train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = torchvision.datasets.MNIST(root=args.input, train=False, transform=transforms.ToTensor(), download=True)
        elif args.dataset == 'FashionMNIST':
            train_dataset = torchvision.datasets.FashionMNIST(root=args.input, train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = torchvision.datasets.FashionMNIST(root=args.input, train=False, transform=transforms.ToTensor(), download=True)
        else:
            raise('{} is not a valid dataset, please choose FashionMNIST\MNIST'.format(args.dataset))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if args.train_vae:
            train_loss_per_epoch = []
            test_loss_per_epoch = []
            results = []
            try:
                for epoch in range(num_epochs):
                    total_loss_train = train_vae(model, train_loader, optimizer, device, args, epoch)
                    avg_loss_train = total_loss_train/len(train_dataset)
                    total_loss_validation = test_vae(model, test_loader, device, args, epoch)
                    avg_loss_test = total_loss_validation/len(test_dataset)

                    print('Epoch {}: mean train loss = {:.3f}, mean validation loss = {:.3f}'.format(
                        epoch + 1, avg_loss_train, avg_loss_test))

                    train_loss_per_epoch.append(avg_loss_train)
                    test_loss_per_epoch.append(avg_loss_test)

            except KeyboardInterrupt:
                print("Training stopped by keyboard interrupt")

            model_name = f'{args.dataset}_VAE_lr_{args.lr}_epochs_{args.epochs}'
            torch.save(model.state_dict(), f"{args.output}/{model_name}.pt")

            results.append({'scenario': model_name, 'train_acc': f'{train_loss_per_epoch[-1]:.3f}',
                            'test_acc': f'{test_loss_per_epoch[-1]:.3f}'})
            df = pd.DataFrame(results)
            df.to_csv(f'{args.output}/results.csv', mode='a', header=not os.path.isfile(f'{args.output}/results.csv'),
                      index='False')

            plt.plot(range(1, num_epochs + 1), torch.Tensor(train_loss_per_epoch).cpu(), label='Train')
            plt.plot(range(1, num_epochs + 1), torch.Tensor(test_loss_per_epoch).cpu(), label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Accuracy vs. Epoch for {model_name}')
            plt.legend()
            if args.show:
                plt.show()
            else:
                plt.savefig(f"{args.output}/{model_name}.png")
        else:
            if args.vae_model_path == None:
                print("please supply vae_model_path (or set --train_vae)")
            else:
                model.load_state_dict(torch.load(args.vae_model_path, map_location=device))

        if args.train_svm:
            num_of_classes = 10
            num_of_train_samples = [100, 600, 1000, 3000]
            if not os.path.exists(os.path.join(args.output, 'svm_models')):
                os.mkdir(os.path.join(args.output, 'svm_models'))
            for n_samples in num_of_train_samples:
                svm_model = train_svm(int(n_samples / num_of_classes), num_of_classes, model, train_loader, 28, 28)
                test_accuracy = test_svm(svm_model, model, test_loader)
                print('n_samples = {}, test accuracy: {:.2f}'.format(n_samples,test_accuracy))
                dump(svm_model, os.path.join(args.output, 'svm_models', f'{n_samples}_samples_{args.dataset}_SVM_kernel_{args.svm_kernel}.joblib'))

    elif args.phase == "test":
        if args.vae_model_path == None:
            raise("Specify trained VAE model path!")
        elif args.svm_models_dir == None:
            raise("Specify trained SVM model path!")
        else:
            if args.dataset == 'FashionMNIST':
                test_dataset = torchvision.datasets.FashionMNIST(root=args.input, train=False, transform=transforms.ToTensor(), download=True)
            elif args.dataset == 'MNIST':
                test_dataset = torchvision.datasets.MNIST(root=args.input, train=False, transform=transforms.ToTensor(), download=True)
            else:
                raise ('{} is not a valid dataset, please choose FashionMNIST\MNIST'.format(args.dataset))

            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            model.load_state_dict(torch.load(args.vae_model_path, map_location=device))
            num_of_train_samples = [100, 600, 1000, 3000]
            svm_models_names = os.listdir(args.svm_models_dir)
            for n_samples in num_of_train_samples:
                svm_model = get_svm_model(args.svm_models_dir, n_samples, args.dataset)
                test_accuracy = test_svm(svm_model, model, test_loader)
                print(f"Test accuracy for {n_samples} samples: {test_accuracy:.2f} for VAE {args.vae_model_path} and SVM {args.svm_models_dir}")

    else:
        raise(f'received invalid phase {args.phase}')


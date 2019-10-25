import torch
from .base_model import BaseModel
from . import networks


class MAGritteModel(BaseModel):
    """ This class implements the MAGritte model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    MAGritte paper:  # TODO
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1_Gb', type=float, default=10.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L1_Gc', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--lambda_M', type=float, default=10.0, help='weight for M loss')
            parser.add_argument('--lambda_Gb_GAN', type=float, default=0.1, help='weight for Gb_GAN loss')

        return parser

    def __init__(self, opt):
        """Initialize the MAGritteModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['Gb_GAN', 'Gb_L1', 'Gb_M', 'Db_real', 'Db_fake', 'Gc_GAN', 'Gc_L1', 'Gc', 'Dc_fake', 'Dc_real']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'fake_B_RGB', 'fake_B_A', 'real_B', 'fake_C', 'fake_C_real_A', 'fake_C_real_B', 'real_C']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['Gb', 'Db', 'Gc', 'Dc']
        else:  # during test time, only load Gb, Gc
            self.model_names = ['Gb', 'Gc']
        # define networks (both generator and discriminator)
        self.netGb = networks.define_G(opt.input_nc, opt.input_nc + 1, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netGc = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netDb = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDc = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_Gb = torch.optim.Adam(self.netGb.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Gc = torch.optim.Adam(self.netGc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Db = torch.optim.Adam(self.netDb.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Dc = torch.optim.Adam(self.netDc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Gb)
            self.optimizers.append(self.optimizer_Gc)
            self.optimizers.append(self.optimizer_Db)
            self.optimizers.append(self.optimizer_Dc)

        self.real_C_real_B = None

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.image_paths = input['A_paths']['A']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B_RGBA = self.netGb(self.real_A)  # Gb(A)
        if self.opt.input_nc == 3:
            self.fake_B_RGB = self.fake_B_RGBA[:,0:3,:,:]
            self.fake_B_A = (self.fake_B_RGBA[:,3,:,:] + 1.0) / 2.0
        else:
            self.fake_B_RGB = self.fake_B_RGBA[:,0:1,:,:]
            self.fake_B_A = (self.fake_B_RGBA[:,1,:,:] + 1.0) / 2.0
        
        self.fake_B = self.real_A * (self.fake_B_A * -1.0 + 1.0) + self.fake_B_RGB * self.fake_B_A
        self.fake_B_A = self.fake_B_A.unsqueeze(0)
        self.fake_C = self.netGc(self.fake_B.detach())  # Gc(fake_B)
        self.fake_C_real_A = self.netGc(self.real_A)  # Gc(real_A)
        self.fake_C_real_B = self.netGc(self.real_B)  # Gc(real_B)
        if self.real_C_real_B is None:
            self.real_C_real_B = self.fake_C_real_B.new_full(self.fake_C_real_B.size(), -1.0)


    def backward_Db(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netDb(self.fake_B.detach())
        self.loss_Db_fake = self.criterionGAN(pred_fake, False)
        # Real
        #real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netDb(self.real_B)
        self.loss_Db_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_Db = (self.loss_Db_fake + self.loss_Db_real) * 0.5
        self.loss_Db.backward()

    def backward_Dc(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_BC = torch.cat((self.fake_B, self.fake_C), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netDc(fake_BC.detach())
        self.loss_Dc_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_BC = torch.cat((self.fake_B, self.real_C), 1)
        pred_real = self.netDc(real_BC.detach())
        self.loss_Dc_real = self.criterionGAN(pred_real, True)
        
        # Fake; stop backprop to the generator by detaching fake_B
        fake_BC = torch.cat((self.real_B, self.fake_C_real_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netDc(fake_BC.detach())
        self.loss_Dc_fake_B = self.criterionGAN(pred_fake, False)
        # Real
        real_BC = torch.cat((self.real_B, self.real_C_real_B), 1)
        pred_real = self.netDc(real_BC.detach())
        self.loss_Dc_real_B = self.criterionGAN(pred_real, True)
        
        # combine loss and calculate gradients
        self.loss_Dc = (self.loss_Dc_fake + self.loss_Dc_real) * 0.25 + (self.loss_Dc_fake_B + self.loss_Dc_real_B) * 0.25
        self.loss_Dc.backward()

    def backward_Gb(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, Gb(A) should fake the discriminator
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netDb(self.fake_B)
        self.loss_Gb_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_Gb_GAN
        # Second, Gb(A) = B
        self.loss_Gb_L1 = self.criterionL1(self.fake_B, self.real_A) * self.opt.lambda_L1_Gb
        # Third, Gc(Gb(A)) -> min
        fake_C = self.netGc(self.fake_B)  # Gc(B)
        #print(fake_C.mean())
        is_fake = ((self.real_C + 1) / 2.0).mean() > 0.001
        self.loss_Gb_M = torch.abs(fake_C.mean() - -1) * self.opt.lambda_M * is_fake
        # combine loss and calculate gradients
        self.loss_Gb = self.loss_Gb_GAN + self.loss_Gb_L1 + self.loss_Gb_M
        #self.loss_Gb = self.loss_Gb_M
        self.loss_Gb.backward()

    def backward_Gc(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, Gc(B) should fake the discriminator
        fake_BC = torch.cat((self.fake_B, self.fake_C), 1)
        pred_fake = self.netDc(fake_BC.detach())
        self.loss_Gc_GAN = self.criterionGAN(pred_fake, True)
        # Second, Gc(B) = C
        self.loss_Gc_L1 = self.criterionL1(self.fake_C, self.real_C) * self.opt.lambda_L1_Gc
        # Third, Gc(real_A) = C
        self.loss_Gc_real_A_L1 = self.criterionL1(self.fake_C_real_A, self.real_C) * self.opt.lambda_L1_Gc
        self.loss_Gc_real_A_L1 = 0
        # Fourth, Gc(real_B) = 0
        self.loss_Gc_real_B_L1 = self.criterionL1(self.fake_C_real_B, self.real_C_real_B) * self.opt.lambda_L1_Gc
        #self.loss_Gc_real_B_L1 *= 0.15
        #print('self.loss_Gc_real_B_L1 = {}'.format(self.loss_Gc_real_B_L1))
        # combine loss and calculate gradients
        self.loss_Gc = self.loss_Gc_GAN + self.loss_Gc_L1 + self.loss_Gc_real_B_L1 + self.loss_Gc_real_A_L1
        self.loss_Gc.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: Gb(A), Gc(B)
        # update Dc
        self.set_requires_grad(self.netDc, True)  # enable backprop for Dc
        self.optimizer_Dc.zero_grad()     # set Dc's gradients to zero
        self.backward_Dc()                # calculate gradients for Dc
        self.optimizer_Dc.step()          # update Dc's weights
        # update Db
        self.set_requires_grad(self.netDb, True)  # enable backprop for Db
        self.optimizer_Db.zero_grad()     # set Db's gradients to zero
        self.backward_Db()                # calculate gradients for Db
        self.optimizer_Db.step()          # update Db's weights
        # update Gc
        self.set_requires_grad(self.netDc, False)  # Dc requires no gradients when optimizing Gc
        self.set_requires_grad(self.netDb, False)  # Db requires no gradients when optimizing Gb
        self.set_requires_grad(self.netGb, False)  # Gc requires no gradients when optimizing Gb
        self.optimizer_Gc.zero_grad()        # set Gb's gradients to zero
        self.backward_Gc()                   # calculate graidents for Gb
        self.optimizer_Gc.step()             # udpate Gb's weights
        # update Gb
        self.set_requires_grad(self.netDc, False)  # Dc requires no gradients when optimizing Gc
        self.set_requires_grad(self.netDb, False)  # Db requires no gradients when optimizing Gb
        self.set_requires_grad(self.netGc, False)  # Gc requires no gradients when optimizing Gb
        self.set_requires_grad(self.netGb, True)  # Gc requires no gradients when optimizing Gb
        self.optimizer_Gb.zero_grad()        # set Gb's gradients to zero
        self.backward_Gb()                   # calculate graidents for Gb
        self.optimizer_Gb.step()             # udpate Gb's weights
        self.set_requires_grad(self.netGc, True)  # Gc requires no gradients when optimizing Gb


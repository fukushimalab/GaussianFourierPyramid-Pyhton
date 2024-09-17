from typing import Self
import cv2
from math import cos, exp, pi, sin, sqrt
from numpy import uint8



@staticmethod
def buildPyramid(src:cv2.Mat, destPyramid:list[cv2.Mat], level:int, borderType=cv2.BORDER_DEFAULT) -> None:
	if src is None or level < 0:
		raise ValueError("Invalid source image or maxlevel.")

	destPyramid = [src]
	current_level = src.copy()
	# for i in range(level):
	# 	current_level = cv2.pyrDown(current_level, borderType = borderType)
	# 	destPyramid.append(current_level)
	for i in range(level):
		temp = cv2.pyrDown(src, dstsize=None, borderType=cv2.BORDER_DEFAULT)
		destPyramid.append(temp)
		current_level = temp


#image input version
@staticmethod 
def buildLaplacianPyramid(src :cv2.Mat, destPyramid :list[cv2.Mat], level :int) -> None:

	if len(destPyramid) != level + 1:
		destPyramid = [0] * (level + 1)

	cv2.buildPyramid(src, destPyramid, level);
	for i in range(level):
		temp :cv2.Mat
		cv2.pyrUp(destPyramid[i + 1], temp, destPyramid[i].shape);
		cv2.subtract(destPyramid[i], temp, destPyramid[i]);

#pyramid input version
@staticmethod
def buildLaplacianPyramid(GaussianPyramid :list[cv2.Mat],destPyramid :list[cv2.Mat],level :int) -> None:
	if len(destPyramid) != level + 1:
		destPyramid = [0] * (level + 1)

	for l in range(level - 2):
		temp :cv2.Mat
		cv2.pyrUp(GaussianPyramid[l + 1], temp, GaussianPyramid[l].shape)
		cv2.subtract(GaussianPyramid[l], temp, destPyramid[l])

@staticmethod
def collapseLaplacianPyramid(LaplacianPyramid :list[cv2.Mat], dest :cv2.Mat) ->  None:
	level : int = len(LaplacianPyramid)
	ret :cv2.Mat
	cv2.pyrUp(LaplacianPyramid[level - 1], ret, LaplacianPyramid[level - 2].shape)
	for i in range(level - 2,0,-1):
		cv2.add(ret, LaplacianPyramid[i], ret)
		cv2.pyrUp(ret, ret, LaplacianPyramid[i - 1].shape)

	cv2.add(ret, LaplacianPyramid[0], dest)



class GaussianFourierLLF:

	def __init__(self):
		self.intensityMin: float = 0.0
		self.intensityMax: float = 255.0
		self.intensityRange: float = 255.0
		self.rangemax: int = 256

		self.T:float = 0.0
		self.alpha:list[float] = []
		self.beta:list[float] = []
		self.omega:list[float] = []

		self.FourierPyramidSin:list[cv2.Mat] = []
		self.FourierPyramidCos:list[cv2.Mat] = []
		self.LaplacianPyramid:list[cv2.Mat] = []
		self.GaussianPyramid:list[cv2.Mat] = []

		self.isAdaptive:bool = False
		self.sigma_range:float = 0.0
		self.boost:float = 1.0
		self.level:int = 0
		self.adaptiveSigmaMap:list[cv2.Mat] = []
		self.adaptiveBoostMap:list[cv2.Mat]= []

	def df(self, x, K, intensityRange, sigma_range) -> float:
		s = sigma_range / intensityRange
		kappa = (2 * K + 1) * pi
		psi = kappa * s / x
		phi = (x - 1.0) / s
		return -kappa * exp(-phi * phi) + psi * psi * exp(-psi * psi)


	def computeT_ClosedForm(self, order:int, sigma_range:float, intensityRange:float) -> float:
		x1 = 1.0
		x2 = 15.0
		loop = 20
		diff = 0
		for i in range(loop):
			x = (x1 + x2) / 2.0
			diff = self.df(x, order, intensityRange, sigma_range)
			if diff >= 0.0:
				x2 = x
			else:
				x1 = x
		return x

	def initRangeFourier(self,order, sigma_range, boost) -> None:

		if len(self.alpha) != order:
			self.alpha = [0] * order #C++版はalpha.resize(order)なので、0で埋めると問題かも
			self.beta = [0] * order

		if len(self.omega) != order:
			self.omega = [0] * order

		T = self.intensityRange * self.computeT_ClosedForm(order, sigma_range, self.intensityRange)
		# Eq. (12), detail information is in K. Sugimoto and S. Kamata, "Compressive bilateral filtering," IEEE Transactions on Image Processing, vol. 24, no. 11, pp.3357-3369, 2015.
	
		#compute omega and alpha in Eqs. (9) and (10)
		for k in range(order):
			self.omega[k] = float(2 * pi / T * (k + 1));
			coeff_kT = self.omega[k] * sigma_range;
			self.alpha[k] = float(2.0 * exp(-0.5 * coeff_kT * coeff_kT) * sqrt(2 * pi) * sigma_range / T);

	def remapCos(self,src:cv2.Mat, dest:cv2.Mat, omega:float) -> None:
		dest= src.copy() * 0
		for y in range(src.shape[0]):
			for x in range(src.shape[1]):
				dest[y][x] = cos(omega * src[y][x])
		print(dest)
		print(omega)

		

	def remapSin(self,src:cv2.Mat, dest:cv2.Mat, omega:float) -> None:
		dest= src.copy() * 0
		for y in range(src.shape[0]):
			for x in range(src.shape[1]):
				dest[y][x] = sin(omega * src[y][x])


	def productSumPyramidLayer(self, srccos:cv2.Mat, srcsin:cv2.Mat, gauss:cv2.Mat, dest:cv2.Mat, omega:float, alpha:float, sigma:float, boost:float) -> None:
		dest = srccos.copy() * 0
		lalpha = -sigma * sigma * omega * alpha * boost
		
		for y in range(srccos.shape[0]):
			for x in range(srccos.shape[1]):
				ms = omega * gauss[y][x]
				dest[y][x] += lalpha * (sin(ms) * srccos[y][x] - cos(ms) * srcsin[y][x])

	def getAdaptiveAlpha(self,coeff:float, base:float, sigma:float, boost:float) -> float:
		a = coeff * sigma
		return sigma * sigma * sigma *boost * base * cv2.exp(-0.5 * a * a)

	def productSumAdaptivePyramidLayer(self, srccos:cv2.Mat ,srcsin:cv2.Mat, gauss:cv2.Mat, dest:cv2.Mat, omega:float, alpha:cv2.Mat,adaptiveSigma:cv2.Mat, adaptiveBoost:cv2.Mat ) -> None:
		dest = srccos.copy() * 0

		base:float = -float(2.0 * cv2.sqrt(2 * pi) * omega / self.T)
		for y in range(srccos.shape[0]):
			for x in range(srccos.shape[1]):
				lalpha = self.getAdaptiveAlpha(omega, base, adaptiveSigma[y][x], adaptiveBoost[y][x])
				ms = omega * gauss[y][x]
				dest[y][x] += lalpha * (sin(ms) * srccos[y][x] - cos(ms) * srcsin[y][x]) 

	def setAdaptive(self, sigmaMap:cv2.Mat, boostMap:cv2.Mat, level:int) -> None:
		self.isAdaptive = True
		buildPyramid(sigmaMap, self.adaptiveSigmaMap, level)
		buildPyramid(boostMap, self.adaptiveBoostMap, level)

	def setFix(self,sigma_range:float, boost:float) -> None:
		self.isAdaptive = False
		self.sigma_range = sigma_range
		self.boost = boost

	#grayscale processing
	def gray(self,src:cv2.Mat, dest:cv2.Mat, order:int, level:int) -> None:
		if len(self.GaussianPyramid) != level + 1:
			self.GaussianPyramid = [0] * (level + 1)
		if len(self.FourierPyramidCos) != level + 1:
			self.FourierPyramidCos = [0] * (level + 1)
		if len(self.FourierPyramidSin) != level + 1:
			self.FourierPyramidSin = [0] * (level + 1)

		if src.dtype == uint8:
			self.GaussianPyramid[0] = src.astype('float32')
		else:
			self.GaussianPyramid[0] = src.copy()

		#compute alpha omega
		self.initRangeFourier(order, self.sigma_range, self.boost)

		#(1) Build Gaussian Pyramid
		#buildPyramid()が使えないようなので以下でpyrDown()で代用 buildPyramid(GaussianPyramid[0], GaussianPyramid, level);
		buildPyramid(self.GaussianPyramid[0], self.GaussianPyramid, level)
		# for i in range(1, level):
		# 	next_level = cv2.pyrDown(current_level)
		# 	self.GaussianPyramid.append(next_level)
		# 	current_level = next_level

		#(2) Build Laplacian Pyramid
		#(2-1) Build Laplacian Pyramid for DC
		buildLaplacianPyramid(self.GaussianPyramid, self.LaplacianPyramid, level);

		for k in range(order):
			#(2-2) Build Remapped Laplacian Pyramid for Cos
			#ここがおかしい
			self.remapCos(self.GaussianPyramid[0], self.FourierPyramidCos[0], self.omega[k]);
			
			#build cos Gaussian pyramid and then generate Laplacian pyramid
			
			buildLaplacianPyramid(self.FourierPyramidCos[0], self.FourierPyramidCos, level);
			
			# (2-3) Build Remapped Laplacian Pyramid for Sin
			self.remapSin(self.GaussianPyramid[0], self.FourierPyramidSin[0], self.omega[k]);
			#build sin Gaussian pyramid and then generate Laplacian pyramid
			buildLaplacianPyramid(self.FourierPyramidSin[0], self.FourierPyramidSin, level);

			#(3) product-sum Gaussian Fourier pyramid
			
			self.LaplacianPyramid = [0] * level
			if self.isAdaptive:
				for l in range(level):
					self.productSumAdaptivePyramidLayer(self.FourierPyramidCos[l],self. FourierPyramidSin[l], self.GaussianPyramid[l], self.LaplacianPyramid[l], self.omega[k], self.alpha[k], self.adaptiveSigmaMap[l], self.adaptiveBoostMap[l]);
			else:
				for l in range(level):
					self.productSumPyramidLayer(self.FourierPyramidCos[l], self.FourierPyramidSin[l], self.GaussianPyramid[l], self.LaplacianPyramid[l], self.omega[k], self.alpha[k], self.sigma_range, self.boost);

		#set last level
		self.LaplacianPyramid[level] = self.GaussianPyramid[level];

		#(4) Collapse Laplacian Pyramid
		collapseLaplacianPyramid(self.LaplacianPyramid, self.LaplacianPyramid[0])

		#convertToがPythonOpenCVにないので、convertScaleAbsで代用　LaplacianPyramid[0].convertTo(dest, src.depth())
		dest = cv2.convertScaleAbs(self.LaplacianPyramid[0]) #C++版ではconvert 32F to output typeだが、ここでは32Fから8Uのみ

	#main processing
	def body(self,src:cv2.Mat, dest:cv2.Mat, order:int, level:int) -> None:
		if src.shape[2] == 1:
			self.gray(src, dest, order, level)
		else:
			onlyY = True
			if onlyY:
				gim = src.copy() * 0
				cv2.cvtColor(src,cv2.COLOR_BGR2YUV,gim)
				vsrc = cv2.split(gim)
				self.gray(vsrc[0], vsrc[0], order, level)
				dest = cv2.merge(vsrc)
				cv2.cvtColor(dest, cv2.COLOR_YUV2BGR, dest)
			else:
				vsrc: list[cv2.Mat]
				vdst: list[cv2.Mat]
				cv2.split(src, vsrc);
				self.gray(vsrc[0], vdst[0], order, level);
				self.gray(vsrc[1], vdst[1], order, level);
				self.gray(vsrc[2], vdst[2], order, level);
				cv2.merge(vdst, dest);

	#public method
	#fixed parameter (sigma_range and boost: same methods: Fast LLF and Fourier LLF)
	def fixed_filter(self,src:cv2.Mat, dest:cv2.Mat, order:int, sigma_range:float, boost:float ,level:int = 2) -> None:
		self.setFix(sigma_range,boost)
		self.body(src, dest, order, level)

	#adaptive parameter (sigma_range and boost: same methods: Fast LLF and Fourier LLF)
	def adaptuve_filter(self,src:cv2.Mat, dest:cv2.Mat, order:int, sigma_range:cv2.Mat, boost:cv2.Mat,level:int = 2) -> None:
		self.setAdaptive(sigma_range, boost, level)
		self.body(src, dest, order, level)

# from pyexpat import native_encoding
# from re import A, X
# from typing import Self
# import cv2

# #image input version
# @staticmethod 
# def buildLaplacianPyramid(SRC :cv2.Mat, destPyramid :list[cv2.Mat], LEVEL :int):

# 	if destPyramid.size() != LEVEL + 1:
# 		destPyramid.resize(LEVEL + 1)

# 	cv2.buildPyramid(SRC, destPyramid, LEVEL);
# 	for i in range(LEVEL):
# 		temp :cv2.Mat
# 		cv2.pyrUp(destPyramid[i + 1], temp, destPyramid[i].size());
# 		cv2.subtract(destPyramid[i], temp, destPyramid[i]);

# #pyramid input version
# @staticmethod
# def buildLaplacianPyramid(GAUSSIANPYRAMID :list[cv2.Mat],destPyramid :list[cv2.Mat],LEVEL :int) -> None:
# 	if destPyramid.size() != LEVEL + 1:
# 		destPyramid.resize(LEVEL + 1)

# 	for l in range(LEVEL - 2):
# 		temp :cv2.Mat
# 		cv2.pyrUp(GAUSSIANPYRAMID[l + 1], temp, GAUSSIANPYRAMID[l].size())
# 		cv2.subtract(GAUSSIANPYRAMID[l], temp, destPyramid[l])

# @staticmethod
# def collapseLaplacianPyramid(LAPLACIANPYRAMID :list[cv2.Mat], dest :cv2.Mat) ->  None:
# 	LEVEL : int = int(LAPLACIANPYRAMID.size())
# 	ret :cv2.Mat
# 	cv2.pyrUp(LAPLACIANPYRAMID[LEVEL - 1], ret, LAPLACIANPYRAMID[LEVEL - 2].size())
# 	for i in range(LEVEL - 2,0,-1):
# 		cv2.add(ret, LAPLACIANPYRAMID[i], ret)
# 		cv2.pyrUp(ret, ret, LAPLACIANPYRAMID[i - 1].size())

# 	cv2.add(ret, LAPLACIANPYRAMID[0], dest)

# # Fast Local Laplacian Filter
# # M. Aubry, S. Paris, J. Kautz, and F. Durand, "Fast local laplacian filters: Theory and applications," ACM Transactionson Graphics, vol. 33, no. 5, 2014.
# # Z. Qtang, L. He, Y. Chen, X. Chen, and D. Xu, "Adaptive fast local laplacian filtersand its edge - aware application," Multimedia Toolsand Applications, vol. 78, pp. 5, 2019.
# class FastLLF:
# 	INTENSITYMIN : float = 0.0
# 	INTENSITYMAX : float = 255.0
# 	INTENSITYRANGE : float = 255.0
# 	RANGEMAX : int = 256

# 	GAUSSIANPYRAMID: list[cv2.Mat]
# 	LAPLACIANPYRAMIDORDER: list[list[cv2.Mat]] #(LEVEL + 1) x order
# 	LAPLACIANPYRAMID: list[cv2.Mat] #LEVEL + 1

# 	isAdaptive: bool = False
# 	sigma_range: float = 0.0 #sigma_r in Eq.6
# 	boost: float = 1.0 #m in Eq.6
# 	adaptiveSigmaMap :cv2.Mat #Sec III.B Pixel-by-pixel enhancement
# 	adaptiveBoostMap :cv2.Mat #Sec III.B Pixel-by-pixel enhancement

# 	#compute interval parameter in linear interpolation (Sec. II.D. Fast Local Laplacian Filtering)
# 	def getTau(K :int, ORDER :int) -> float:
# 		global INTENSITYRANGE,INTENSITYMIN
# 		DELTA: float = INTENSITYRANGE / (ORDER - 1)
# 		return float(K * DELTA + INTENSITYMIN)

# 	def remap(SRC: cv2.Mat, dest: cv2.Mat, G: float, SIGMA_RANGE: float, BOOST: float) -> None:
# 		if SRC.data != dest.data:
# 			dest.create(SRC.size(), cv2.CV_32F)

# 		SIZE: int = SRC.size().area()
# 		#•Û—¯
# 		S: float
# 		d: float
# 		COEFF: float = float(1.0 / (-2.0 * SIGMA_RANGE * SIGMA_RANGE))
		
# 		for i in range(SIZE):
# 			X: float = S[i] - G
# 			d[i] = X * BOOST * cv2.exp(X * X * COEFF) + S[i]

# 	def remapAdaptive(SRC: cv2.Mat, dest: cv2.Mat, G: float, SIGMA_RANGE: cv2.Mat, BOOST: cv2.Mat) -> None:
# 		if SRC.data != dest.data:
# 			dest.create(SRC.size(), cv2.CV_32F)

# 		SIZE: int = SRC.size().area()
# 		#•Û—¯
# 		S: float
# 		d: float
# 		SIGMAPTR: float
# 		BOOSTPTR: float

# 		for i in range(SIZE):
# 			COEFF: float = 1.0 / (-2.0 * SIGMAPTR[i] * SIGMAPTR[i]);
# 			BOOST: float = BOOSTPTR[i];
# 			X: float = S[i] - G;
# 			d[i] = X * BOOST * cv2.exp(X * X * COEFF) + S[i];

# 	def getLinearIndex(v: float, index_l: int, index_h: int, alpha: float, ORDER: int, INTENSITYMIN: float, INTENSITYMAX: float) -> None:
# 		INTENSITYRANGE: float = INTENSITYMAX - INTENSITYMIN
# 		DELTA: float = INTENSITYRANGE / (ORDER - 1)
# 		I: int = int(v / DELTA)

# 		if I < 0:
# 			index_l = 0
# 			index_h = 0
# 			alpha = 1.0
# 		elif I + 1 > ORDER - 1:
# 			index_l = ORDER - 1
# 			index_h = ORDER - 1
# 			alpha = 0.0
# 		else:
# 			index_l = I
# 			index_h = I + 1
# 			alpha = 1.0 - (v - (I * DELTA)) / DELTA

# 	#last level is not blended; thus, inplace operation for input Gaussian Pyramid is required.
# 	def blendLaplacianLinear(LAPLACIANPYRAMID: list[list[cv2.Mat]], GaussianPyramid: list[cv2.Mat], destPyramid: list[cv2.Mat], ORDER: int) -> None:
# 		LEVEL: int = int(GaussianPyramid.size())
# 		destPyramid.resize(LEVEL)
# 		#•Û—¯
# 		#std::vector<const float*> lptr(order);
# 		for l in range(LEVEL - 1):
# 			destPyramid[l].create(GaussianPyramid[l].size(),cv2.CV_32F)
# 			#•Û—¯
# 			g = GaussianPyramid[l]
# 			d = destPyramid[l]
# 			for k in range(ORDER):
# 				#•Û—¯
# 				dammy = 0

# 			for i in range(GaussianPyramid.size().area()):
# 				alpha: float
# 				high: int
# 				low: int
# 				getLinearIndex(g[i],low,high,alpha,ORDER,INTENSITYMIN,INTENSITYMAX)
# 				d[i] = alpha * lptr[low][i] + (1.0 + alpha) * lptr[high][i]

# 	def setAdaptive(SIGMAP: cv2.Mat, BOOSTMAP: cv2.Mat) -> None:
# 		isAdaptive = True
# 		global adaptiveSigmap
# 		global adaptiveBoosrMap
# 		adaptiveSigmap = SIGMAP
# 		adaptiveBoosrMap = BOOSTMAP

# 	def setFix(SIGMA_RANGE: float,BOOST: float) -> None:
# 		isAdaptive = False
# 		global sigma_range
# 		global boost
# 		sigma_range = SIGMA_RANGE
# 		boost = BOOST

# 	#grayscale processing
# 	def gray(SRC: cv2.Mat, dest: cv2.Mat, ORDER: int,LEVEL: int) -> None:
# 		#1 alloc
# 		if GaussianPyramid.size() != LEVEL + 1:
# 			GaussianPyramid.resize(LEVEL + 1)

# 		LaplacianPyramidOrder.resize(ORDER)

# 		for n in range(ORDER):
# 			LaplacianPyramidOrder[n].resize(LEVEL + 1)

# 		if SRC.depth() == cv2.CV_32F:
# 			SRC.copyTo(GaussianPyramid[0])
# 		else:
# 			S
			 

# 	#main processing (same methods: Fast LLF and Fourier LLF)
# 	def body() -> None:
# 		dammy = 1
# 	#fix parameter (sigma_range and boost)
# 	def filter() -> None:
# 		dammy = 1

# 	#adaptive parameter (sigma_range and boost)
# 	def filter() -> None:
# 		dammy = 1

# #Fourier Local Laplacian Filter
# #Y. Sumiya, T. Otsuka, Y. Maedaand N. Fukushima, "Gaussian Fourier Pyramid for Local Laplacian Filter," IEEE Signal Processing Letters, vol. 29, pp. 11-15, 2022.
# class GaussianFourierLLF:
# 	INTENSITYMIN: float = 0.0
# 	INTENSITYMAX: float = 255.0
# 	INTENSITYRANGE: float = 255.0
# 	RANGEMAX: int = 256

	
# 	def df(x: float,K: int, IRANGE: float,SIGMA_RANGE: float) -> float:
# 		S: float = SIGMA_RANGE / IRANGE
# 		KAPPA: float = (2 * K + 1) * cv2.CV_PI
# 		PSI: float = KAPPA * S / X
# 		PHI: float = (X - 1.0) / S
# 		return (-KAPPA * cv2.exp(-PHI * PHI) + PSI * PSI * cv2.exp(-PSI * PSI))

# 	def computeT_ClosedForm(order: int, sigma_range: float,INTESITYRANGE: float) -> float:
# 		x:float
# 		diff:float

# 		x1:float = 1.0
# 		x2:float = 15.0
# 		loop:int = 20
# 		for i in range(loop):
# 			x = (x1 + x2) / 2.0
# 			diff = self.df(x, order,INTENSITYRANGE,sigma_range)
# 			if 0.0 <= diff:
# 				x2 = x
# 			else:
# 				x1 = x
			
# 			return x



		


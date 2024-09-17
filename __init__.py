import cv2 #ver4.10.0
import LLFhpp

# alpha blending comparison between src1 and src2 by GUI
def compare(WNAME: str, SRC1: cv2.Mat, SRC2: cv2.Mat):
    cv2.namedWindow(WNAME)
    a: int = 0
    cv2.createTrackbar("alpha", WNAME, a, 100)
    key: int = 0
    show: cv2.Mat

    while key != 'q':
        cv2.addWeighted(SRC1, a * 0.01, SRC2, (100 - a) * 0.01, 0.0, show)
        cv2.imshow(WNAME, show)
        key = cv2.waitKey(1)

def main() -> int:
    #load image
    src = cv2.imread("flower.png")
    #destination image
    destFourierLLF = src.copy() * 0
    #destFastLLFAaptive :cv2.Mat 
    destFourierLLFAaptive = src.copy() * 0

    #parameter setting
    SIGMA: float = 30.0
    BOOST: float = 2.0
    LEVEL = 2
    ORDER = 4

    #create instance
    #llf: LLFhpp.FastLLF
    gfllf = LLFhpp.GaussianFourierLLF()
 
    #parameter fix filter
    #llf.filter(src, destFastLLF, ORDER * 2, SIGMA, BOOST, LEVEL); #order*2: FourierLLF requires double pyramids due to cos and sin pyramids; thus we double the order to adjust the number of pyramids.
    gfllf.fixed_filter(src, destFourierLLF, ORDER, SIGMA, BOOST, LEVEL);

    #parameter adaptive filter
    #generate parameter maps (circle shape)
    sigmaMap = src.copy() #Mat sigmaMap(src.size(), CV_32F);
    sigmaMap.setTo(SIGMA);
    
    cv2.circle(sigmaMap, cv2.Point(src.size()) // 2, src.cols // 4, cv2.Scalar.all(SIGMA * 2.0), cv2.FILLED )

    boostMap = src.copy() #Mat boostMap(src.size(), CV_32F);
    boostMap.setTo(BOOST)
    cv2.circle(boostMap, cv2.Point(src.size()) // 2, src.cols // 4, cv2.Scalar.all(BOOST * 2.0), cv2.FILLED)
   
    #filter
    #llf.filter(src, destFastLLFAaptive, ORDER * 2, sigmaMap, boostMap, LEVEL);
    gfllf.adaptive_filter(src, destFourierLLFAaptive, ORDER, sigmaMap, boostMap, LEVEL);

    cv2.imshow("src", src);
    #cv2.imshow("Fast LLF dest", destFastLLF);
    cv2.imshow("Fourier LLF dest", destFourierLLF);
    #cv2.imshow("Fast LLF Adaptive dest", destFastLLFAaptive);
    cv2.imshow("Fourier LLF Adaptive dest", destFourierLLFAaptive);
    #compare("LLF", destFastLLF, destFourierLLF);#quit `q` key
    #compare("LLF", destFastLLFAaptive, destFourierLLFAaptive);#quit `q` key

    cv2.waitKey(0)

    return 0


if __name__ == "__main__":
    main()
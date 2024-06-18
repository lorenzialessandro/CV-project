import sys
import os

if __name__ == '__main__':

    nargs = len(sys.argv)
    if nargs != 2 :
        print("Arguments error")
        print("Expected argument : 1 mandatory")
        print("     Arg1 : { RENDER | SKINNING }")
        sys.exit(1)
        
    opt = sys.argv[1]
    
    if opt is not None :
        if opt == "RENDER":
            os.chdir("lib/deep-motion-editing/blender_rendering/")
            os.system("blender -P render.py -- --bvh_path ../../../resources/360fps/animation_small.bvh")
        elif opt == "SKINNING":
            os.system("blender -P lib/deep-motion-editing/blender_rendering/skinning.py -- --bvh_file resources/360fps/animation.bvh --fbx_file resources/ue5/TheBoss.fbx")
        else :
            print("Unkown option")
            exit()



        

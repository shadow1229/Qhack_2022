#! /usr/bin/python3

import sys
import numpy as np


def givens_rotations(a, b, c, d):
    """Calculates the angles needed for a Givens rotation to out put the state with amplitudes a,b,c and d

    Args:
        - a,b,c,d (float): real numbers which represent the amplitude of the relevant basis states (see problem statement). Assume they are normalized.

    Returns:
        - (list(float)): a list of real numbers ranging in the intervals provided in the challenge statement, which represent the angles in the Givens rotations,
        in order, that must be applied.
    """

    # QHACK #
    cos_theta_1_sgn = 1
    sin_theta_1_sgn = 1
    
    if b <0:
        sin_theta_1_sgn = 1
    else:
        sin_theta_1_sgn = -1        

        
    theta_2_half = np.arctan2( c * sin_theta_1_sgn, -b*sin_theta_1_sgn )
    theta_2 = 2*theta_2_half
    if theta_2 > (np.pi ):
        theta_2 -= 2*np.pi

    if d < 0:
        sin_theta_3_sgn = 1
    else:
        sin_theta_3_sgn = -1
                
    theta_3_half = np.arctan2(-d*sin_theta_3_sgn , a) * sin_theta_3_sgn 
    theta_3 = 2 * theta_3_half 
    if theta_3 > (np.pi ):
        theta_3 -= 2*np.pi
        
    if (a*np.cos(theta_3 / 2)) <0:
        cos_theta_1_sgn = -1
    
    #print(cos_theta_1_sgn, sin_theta_1_sgn)
    theta_1_half  = np.arctan2( np.sqrt(b**2+c**2) , np.sqrt(1-b**2-c**2)) * sin_theta_1_sgn
      
    theta_1 = 2*(theta_1_half)
                       
    if theta_1 > (np.pi ):
        theta_1 -= 2*np.pi
    return theta_1, theta_2, theta_3
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    theta_1, theta_2, theta_3 = givens_rotations(
        float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3])
    )
    print(*[theta_1, theta_2, theta_3], sep=",")

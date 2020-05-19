#
# Factory function to make a ksfdMat. Do it this way rather than
# instantiatign the type directly, to avoid callign petsc4py.init,
# which happens automatically when ksfdMat is imported. 
#
def getMat(mat):
    """
    Factory function to create a ksfdMat
    """
    from ksfdMat import ksfdMat
    return ksfdMat(mat)

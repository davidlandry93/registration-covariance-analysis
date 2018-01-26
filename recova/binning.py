
import json
import subprocess

from recova.util import eprint

class BinningAlgorithm:
    """A binning alogrithm takes a pointcloud and puts it into bins we can compute a descriptor."""
    def __init__(self):
        pass

    def compute(self, reading, reference):
        raise NotImplementedError('Binning algorithms must implement method compute')

    def __repr__(self):
        raise NotImplementedError('BinningAlgorithms must implement __repr__')


class CylindricalBinningAlgorithm(BinningAlgorithm):
    def __init__(self, spanr, spanz, nr, nz, ntheta):
        self.spanr = spanr
        self.spanz = spanz
        self.nr = nr
        self.nz = nz
        self.ntheta = ntheta



class GridBinningAlgorithm(BinningAlgorithm):
    def __init__(self, spanx, spany, spanz, nx, ny, nz):
        self.spanx = spanx
        self.spany = spany
        self.spanz = spanz
        self.nx = nx
        self.ny = ny
        self.nz = nz


    def compute(self, pointcloud):
        command_string = 'grid_pointcloud_separator -spanx {} -spany {} -spanz {} -nx {} -ny {} -nz {}'.format(
            self.spanx,
            self.spany,
            self.spanz,
            self.nx,
            self.ny,
            self.nz
        )
        eprint(command_string)

        response = subprocess.check_output(
            command_string,
            universal_newlines=True,
            shell=True,
            input=json.dumps(pointcloud)
        )

        return json.loads(response)

    def __repr__(self):
        return 'grid-{:.4f}-{:.4f}-{:.4f}-{}-{}-{}'.format(self.spanx, self.spany, self.spanz, self.nx, self.ny, self.nz)

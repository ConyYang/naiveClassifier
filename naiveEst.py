import numpy


class naiveEst:
    def __init__(self, rd_f, wr_f, h = 2):
        self.rd_f = rd_f
        self.wr_f = wr_f
        self.h = h

    def _readfile_(self):
        '''
        Read file
        '''
        file = open(self.rd_f, 'r')
        line = file.readline()
        file.close()

        #retrieve value n & m
        meta_info = line[0].split(",")
        self.n = int(meta_info[0])
        self.m = int(meta_info[1])

        #retrieve each line value
        self.instance_array = []
        for i in range (1, len(line)):
            columnn = line[i].split()
            row = [float(j) for j in columnn]
            self.instance_array.append(self.instance_array)

        self.instance_np = numpy.array(self.instance_array)

    def _window (self, u):
        if ((abs(u) < 0.5).all()):
            return 1
        else:
            return 0

    def _density(self, x):
        p = 0
        V = self.h ** self.d
        for x_i in self.instance_np:
            u = (x - x_i) / self.h
            p += self._window(u)
        return p

    def _estimator(self, width = 2):
        P = []
        self._readfile_()
        for row in self.instance_np:
            P.append(numpy.round(self._density(row, width)))
        self._writefile_(P)

    def _writefile_(self, P):
        file = open(self.wr_f, 'w+')
        for p in P:
            file.write("{}\n".format(p))
        file.close()








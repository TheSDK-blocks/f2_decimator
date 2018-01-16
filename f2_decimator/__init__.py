# f2_decimator class 
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 16.01.2018 12:02
import sys
import os
import numpy as np
import scipy.signal as sig
import tempfile
import subprocess
import shlex
import time
#Add TheSDK to path. Importing it first adds the rest of the modules
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))
from thesdk import *
from refptr import *
from verilog import *
from halfband import *
from cic3 import *

#Simple buffer template
class f2_decimator(verilog,thesdk):
    def __init__(self,*arg): 
        self.proplist = [ ' '];    #properties that can be propagated from parent
        self.Rs_high = 8*160;                      # sampling frequency
        self.Rs_low=20e6
        self.BB_bandwidth=0.45
        self.iptr_A = refptr();
        self.model='py';             #can be set externally, but is not propagated
        self._filters = [];
        self._Z = refptr();
        self._classfile=os.path.dirname(os.path.realpath(__file__)) + "/"+__name__
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
        self.init()

    def init(self):
        self.def_verilog()
        #self._vlogparameters=dict([ ('g_rs',self.Rs_high), ('g_Rs_slow',self.cic3Rs_high_slow), ('g_integscale',self.integscale) ])

    def main(self):
        if self.model=='py':
            print('Generating')
            self.generate_decimator()
            for i in self._filters:
                print('In main')
                i.run()
            out=self._filters[3]._Z.Value

        if self.par:
            queue.put(out)
        self._Z.Value=out

    def run(self,*arg):
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            queue=arg[0]       #multiprocessing.Queue as the first argument
        else:
            self.par=False

        #Example of how to use Python models for sub-blocks, but
        #Merged verilog for the current modeule
        if self.model=='py':
            print('Running main')
            self.main()
        else: 
          self.write_infile()
          self.run_verilog()
          self.read_outfile()

    def generate_decimator(self,**kwargs):
       n=kwargs.get('n',np.array([6,8,40]))
       #if Int(self.Rs_high/self.Rs_low)%2 >0:
       #    self.print_log({'type':'F', 'msg': 'Decimaton ratio must be a multiple of 2')
       self._filters=[cic3()]
       self._filters[0].Rs_high=self.Rs_high
       self._filters[0].Rs_low=self.Rs_low*8
       self._filters[0].iptr_A=self.iptr_A
       self._filters[0].init()

       for i in range(3):
           h=halfband()
           h.halfband_Bandwidth=self.BB_bandwidth/(2**(2-i))
           h.halfband_N=n[i]
           h.Rs_high=self._filters[0].Rs_low/(2**i)
           h.init()
           #h.export_scala()
           self._filters.append(h)
           print(self._filters[i+1].Rs_low)
       #Two ways to do it.
       #lambda can not contain assignment
       #map( lambda prev,next: next.iptr_A=prev._Z, list(zip(self._filters[0:-1], self._filters[1:]))) 
       for i in range(len(self._filters)-1):
           self._filters[i+1].iptr_A=self._filters[i]._Z

       #_=[ i.init() for i in self._filters]
       #for i in self._filters:
       #    print('Initing %s' %(i))
       #    i.init()
       #    print(i.Rs_low)
       
    def write_infile(self):
        rndpart=os.path.basename(tempfile.mkstemp()[1])
        if self.model=='sv':
            self._infile=self._vlogsimpath +'/A_' + rndpart +'.txt'
            self._outfile=self._vlogsimpath +'/Z_' + rndpart +'.txt'
        elif self.model=='vhdl':
            pass
        else:
            pass
        try:
          os.remove(self._infile)
        except:
          pass
        fid=open(self._infile,'wb')
        np.savetxt(fid,self.iptr_A.Value.reshape(-1,1).view(float),fmt='%i', delimiter='\t')
        fid.close()

    def read_outfile(self):
        fid=open(self._outfile,'r')
        out = np.loadtxt(fid,dtype=complex)
        #Of course it does not work symmetrically with savetxt
        out=(out[:,0]+1j*out[:,1]).reshape(-1,1) 
        fid.close()
        if self.par:
          queue.put(out)
        self._Z.Value=out
        os.remove(self._outfile)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from f2_decimator import *
    from  f2_signal_gen import *
    from  f2_system import *
    t=thesdk()
    fsorig=20e6
    highrate=16*8*fsorig
    bw=0.45
    integscale=4096
    siggen=f2_signal_gen()
    fsindexes=range(1,int(highrate/fsorig))
    print(list(fsindexes))
    freqlist=[1.0e6, 0.45*fsorig]
    _=[freqlist.extend([i*fsorig-bw*fsorig, i*fsorig+bw*fsorig]) for i in list(fsindexes) ] 
    print(freqlist)
    siggen.Rs=highrate
    siggen.bbsigdict={ 'mode':'sinusoid', 'freqs':freqlist, 'length':2**20, 'BBRs':highrate };
    siggen.Users=1
    siggen.Txantennas=1
    siggen.init()
    #Mimic ADC This is very ideal to verify frequanyuresponses
    bits=20
    insig=siggen._Z.Value[0,:,0].reshape(-1,1)
    insig=np.round(insig/np.amax([np.abs(np.real(insig)), np.abs(np.imag(insig))])*(2**(bits-1)-1))
    str="Input signal range is %i" %((2**(bits-1)-1))
    t.print_log({'type':'I', 'msg': str})
    h=f2_decimator()
    h.Rs_high=highrate
    h.Rs_low=fsorig
    h.integscale=integscale 
    h.iptr_A.Value=insig.reshape((-1,1))
    h.model='py'
    h.init()
    print('Start running')
    h.run() 


    for filtno in range(len(h._filters)):
        impulse=np.r_['0', h._filters[filtno].H, np.zeros((1024-h._filters[filtno].H.shape[0],1))]
        w=np.arange(1024)/1024*h._filters[filtno].Rs_high
        spe1=np.fft.fft(impulse,axis=0)
        f=plt.figure(3*filtno+1)
        plt.plot(w,20*np.log10(np.abs(spe1)/np.amax(np.abs(spe1))))
        plt.ylim((-80,3))
        plt.grid(True)
        f.show()

        fs, spe2=sig.welch(h._filters[filtno].iptr_A.Value,fs=h._filters[filtno].Rs_high,nperseg=1024,return_onesided=False,scaling='spectrum',axis=0)
        #w=np.arange(spe2.shape[0])/spe2.shape[0]*h._filters[filtno].cic3Rs_slow
        print(h._filters[filtno].Rs_high)
        w=np.arange(spe2.shape[0])/spe2.shape[0]*h._filters[filtno].Rs_high
        ff=plt.figure(3*filtno+2)
        plt.plot(w,10*np.log10(np.abs(spe2)/np.amax(np.abs(spe2))))
        plt.ylim((-80,3))
        plt.grid(True)
        ff.show()
        
        #spe3=np.fft.fft(h._Z.Value,axis=0)
        maximum=np.amax([np.abs(np.real(h._filters[filtno]._Z.Value)), np.abs(np.imag(h._filters[filtno]._Z.Value))])
        str="Output signal range is %i" %(maximum)
        t.print_log({'type':'I', 'msg': str})
        fs, spe3=sig.welch(h._filters[filtno]._Z.Value,fs=h._filters[filtno].Rs_low,nperseg=1024,return_onesided=False,scaling='spectrum',axis=0)
        #w=np.arange(spe3.shape[0])/spe3.shape[0]*h._filters[filtno].cic3Rs_slow
        print(h._filters[filtno].Rs_low)
        w=np.arange(spe3.shape[0])/spe3.shape[0]*h._filters[filtno].Rs_low
        fff=plt.figure(3*filtno+3)
        plt.plot(w,10*np.log10(np.abs(spe3)/np.amax(np.abs(spe3))))
        plt.ylim((-80,3))
        plt.grid(True)
        fff.show()

        #Required to keep the figures open
    input()


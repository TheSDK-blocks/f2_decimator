# f2_decimator class 
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 03.09.2018 19:28
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
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.proplist = [ ' '];    #properties that can be propagated from parent
        self.Rs_high = 8*160e6;                      # sampling frequency
        self.Rs_low=20e6
        self.BB_bandwidth=0.45
        self.iptr_A = IO();
        self.model='py';             #can be set externally, but is not propagated
        self.export_scala=False
        self.scales=[1,1,1,1]
        self.cic3shift=0
        self._filters = [];
        self._Z = IO();
        self.zeroptr=IO()
        self.zeroptr.Data=np.zeros((1,1))
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
        self.init()

    def init(self):
        self.mode=self.determine_mode()                 
        self._vlogmodulefiles=list(["clkdiv_n_2_4_8.sv"])                 
        self._vlogparameters=dict([ ('g_Rs_high',self.Rs_high), ('g_Rs_low',self.Rs_low), 
            ('g_scale0',self.scales[0]),  
            ('g_scale1',self.scales[1]),  
            ('g_scale2',self.scales[2]),  
            ('g_scale3',self.scales[3]),
            ('g_cic3shift',self.cic3shift),
            ('g_mode',self.mode)
            ])

    def main(self):
        if self.mode>0:
            self.generate_decimator()
            for i in range(len(self._filters)):
                self._filters[i].run()
                self._filters[i]._Z.Data=(self._filters[i]._Z.Data*self.scales[i]).reshape(-1,1)
            out=self._filters[-1]._Z.Data
        else:
            out=self.iptr_A.Data
        if self.par:
            self.queue.put(out)
        maximum=np.amax([np.abs(np.real(out)), np.abs(np.imag(out))])
        str="Output signal range is %i" %(maximum)
        self.print_log(type='I', msg=str)
        self._Z.Data=out

    def run(self,*arg):
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]       #multiprocessing.Queue as the first argument
        else:
            self.par=False

        #Example of how to use Python models for sub-blocks, but
        #Merged verilog for the current modeule
        if self.model=='py':
            self.main()
        else: 
          self.write_infile()
          self.run_verilog()
          self.read_outfile()

    def generate_decimator(self,**kwargs):
       n=kwargs.get('n',np.array([6,8,40]))
       self._filters=[]
       for i in range(4-self.mode,4):
           if i==0:
               h=cic3()
               h.Rs_high=self.Rs_high
               h.Rs_low=self.Rs_low*8
               h.integscale=self.scales[0]
               h.cic3shift=self.cic3shift
               h.init()
           else:
               h=halfband()
               h.halfband_Bandwidth=self.BB_bandwidth/(2**(2-i+1))
               h.halfband_N=n[i-1]
               h.Rs_high=self.Rs_low*(2**(4-i))
               h.init()
               if self.export_scala:
                   h.export_scala()
           self._filters.append(h)
       #Here, in order to model multiplier, we would need to 
       # Create multiplier instance with pointers
       for i in range(len(self._filters)):
           if i==0:
               self._filters[i].iptr_A=self.iptr_A
           else:
               self._filters[i].iptr_A=self._filters[i-1]._Z
           self._filters[i].init()

    def determine_mode(self):
        #0=bypass, 1 decimate by 2, 2 decimate by 4, 3 decimate by 8, 4, decimate by more
        M=self.Rs_high/self.Rs_low
        if (M%8!=0) and (M!=4) and (M!=2) and (M!=1):
            self.print_log(type='F', msg="Decimatio ratio is not valid. Must be 1,2,4,8 or multiple of 8")
        else:
            if M<=8:
                mode=int(np.log2(M))
            else:
             mode=int(4)
        self.print_log(type='I', msg="Decimation ratio is set to %i corresponding to mode %i" %(M,mode))
        return mode
       
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
        np.savetxt(fid,self.iptr_A.Data.reshape(-1,1).view(float),fmt='%i', delimiter='\t')
        fid.close()

    def read_outfile(self):
        fid=open(self._outfile,'r')
        out = np.loadtxt(fid,dtype=complex)
        #Of course it does not work symmetrically with savetxt
        out=(out[:,0]+1j*out[:,1]).reshape(-1,1) 
        maximum=np.amax([np.abs(np.real(out)), np.abs(np.imag(out))])
        str="Output signal range is %i" %(maximum)
        self.print_log(type='I', msg=str)
        self._Z.Data=out
        fid.close()
        if self.par:
          self.queue.put(out)
        self._Z.Data=out
        os.remove(self._outfile)

if __name__=="__main__":
    import sys
    import matplotlib.pyplot as plt
    from f2_decimator import *
    from  f2_signal_gen import *
    from  f2_system import *
    arguments=sys.argv[1:]
    t=thesdk()
    fsorig=20e6
    highrate=16*8*fsorig
    bw=0.45
    siggen=f2_signal_gen()
    fsindexes=range(1,3)
    print(list(fsindexes))
    freqlist=[1.0e6, 0.45*fsorig]
    _=[freqlist.extend([i*fsorig-bw*fsorig, i*fsorig+bw*fsorig]) for i in list(fsindexes) ] 
    print(freqlist)
    siggen.Rs=highrate
    siggen.bbsigdict={ 'mode':'sinusoid', 'freqs':freqlist, 'length':2**20, 'BBRs':highrate };
    siggen.Users=1
    siggen.Txantennas=1
    siggen.init()
    #Mimic ADC This is very ideal to verify frequencyuresponses
    bits=10
    insig=siggen._Z.Data[0,:,0].reshape(-1,1)
    insig=np.round(insig/np.amax([np.abs(np.real(insig)), np.abs(np.imag(insig))])*(2**(bits-1)-1))
    str="Input signal range is %i" %((2**(bits-1)-1))
    t.print_log({'type':'I', 'msg': str})
    h=f2_decimator()
    h.Rs_high=highrate
    h.Rs_low=fsorig
    #integscale=np.cumsum(np.cumsum(np.cumsum(np.ones((np.round(h.Rs_high/(8*h.Rs_low)).astype(int),1))*2**(bits-1))))[-1]
    integscale=128
    cic3shift=0
    h.scales=[integscale,1,1,1] 
    h.iptr_A.Data=insig.reshape((-1,1))
    h.export_scala=False
    if len(arguments) >0:
        #h.model='\'%s\'' %(arguments[0])
        h.model='sv'
        print(h.model)
    else:
        h.model='py'
        print(h.model)
    h.init()
    h.run() 

    if h.mode==0 and h.model=='py':
            fs, spe2=sig.welch(h.iptr_A.Data,fs=h.Rs_high,nperseg=1024,return_onesided=False,scaling='spectrum',axis=0)
            w=np.arange(spe2.shape[0])/spe2.shape[0]*h.Rs_high
            ff=plt.figure(0)
            plt.plot(w,10*np.log10(np.abs(spe2)/np.amax(np.abs(spe2))))
            plt.ylim((-80,3))
            plt.grid(True)
            ff.show()

            maximum=np.amax([np.abs(np.real(h._Z.Data)), np.abs(np.imag(h._Z.Data))])
            str="Output signal range is %i" %(maximum)
            t.print_log({'type':'I', 'msg': str})
            fs, spe3=sig.welch(h._Z.Data,fs=h.Rs_low,nperseg=1024,return_onesided=False,scaling='spectrum',axis=0)
            print(h.Rs_low)
            w=np.arange(spe3.shape[0])/spe3.shape[0]*h.Rs_low
            fff=plt.figure(1)
            plt.plot(w,10*np.log10(np.abs(spe3)/np.amax(np.abs(spe3))))
            plt.ylim((-80,3))
            plt.grid(True)
            fff.show()
    elif h.mode>0 and h.model=='py':
        for filtno in range(len(h._filters)):
            impulse=np.r_['0', h._filters[filtno].H, np.zeros((1024-h._filters[filtno].H.shape[0],1))]
            w=np.arange(1024)/1024*h._filters[filtno].Rs_high
            spe1=np.fft.fft(impulse,axis=0)
            f=plt.figure(3*filtno+1)
            plt.plot(w,20*np.log10(np.abs(spe1)/np.amax(np.abs(spe1))))
            plt.ylim((-80,3))
            plt.grid(True)
            f.show()

            fs, spe2=sig.welch(h._filters[filtno].iptr_A.Data,fs=h._filters[filtno].Rs_high,nperseg=1024,return_onesided=False,scaling='spectrum',axis=0)
            w=np.arange(spe2.shape[0])/spe2.shape[0]*h._filters[filtno].Rs_high
            ff=plt.figure(3*filtno+2)
            plt.plot(w,10*np.log10(np.abs(spe2)/np.amax(np.abs(spe2))))
            plt.ylim((-80,3))
            plt.grid(True)
            ff.show()
            
            #spe3=np.fft.fft(h._Z.Data,axis=0)
            maximum=np.amax([np.abs(np.real(h._filters[filtno]._Z.Data)), np.abs(np.imag(h._filters[filtno]._Z.Data))])
            str="Output signal range is %i" %(maximum)
            t.print_log({'type':'I', 'msg': str})
            fs, spe3=sig.welch(h._filters[filtno]._Z.Data,fs=h._filters[filtno].Rs_low,nperseg=1024,return_onesided=False,scaling='spectrum',axis=0)
            #w=np.arange(spe3.shape[0])/spe3.shape[0]*h._filters[filtno].cic3Rs_slow
            print(h._filters[filtno].Rs_low)
            w=np.arange(spe3.shape[0])/spe3.shape[0]*h._filters[filtno].Rs_low
            fff=plt.figure(3*filtno+3)
            plt.plot(w,10*np.log10(np.abs(spe3)/np.amax(np.abs(spe3))))
            plt.ylim((-80,3))
            plt.grid(True)
            fff.show()

            #Required to keep the figures open
    else:
            fs, spe2=sig.welch(h.iptr_A.Data,fs=h.Rs_high,nperseg=1024,return_onesided=False,scaling='spectrum',axis=0)
            #w=np.arange(spe2.shape[0])/spe2.shape[0]*h._filters[filtno].cic3Rs_slow
            w=np.arange(spe2.shape[0])/spe2.shape[0]*h.Rs_high
            ff=plt.figure(1)
            plt.plot(w,10*np.log10(np.abs(spe2)/np.amax(np.abs(spe2))))
            plt.ylim((-80,3))
            plt.grid(True)
            ff.show()
            
            print(h._Z.Data.shape)
            fs, spe3=sig.welch(h._Z.Data,fs=h.Rs_low,nperseg=1024,return_onesided=False,scaling='spectrum',axis=0)
            print(h.Rs_low)
            w=np.arange(spe3.shape[0])/spe3.shape[0]*h.Rs_low
            fff=plt.figure(2)
            plt.plot(w,10*np.log10(np.abs(spe3)/np.amax(np.abs(spe3))))
            plt.ylim((-80,3))
            plt.grid(True)
            fff.show()

    input()


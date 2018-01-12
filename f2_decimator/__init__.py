# f2_decimator class 
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 11.01.2018 18:21
import sys
import os
import numpy as np
import scipy.signal as sig
import tempfile
import subprocess
import shlex
import time
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *
from refptr import *
from rtl import *
from halfband import *

#Simple buffer template
class f2_decimator(rtl,thesdk):
    def __init__(self,*arg): 
        self.proplist = [ 'Rs' ];    #properties that can be propagated from parent
        self.Rs = 1;                 # sampling frequency
        self.iptr_A = refptr();
        self.model='py';             #can be set externally, but is not propagated
        self._Z = refptr();
        self._classfile=os.path.dirname(os.path.realpath(__file__)) + "/"+__name__
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
        self.init()

    def init(self):
        self.def_rtl()
        rndpart=os.path.basename(tempfile.mkstemp()[1])
        self._infile=self._rtlsimpath +'/A_' + rndpart +'.txt'
        self._outfile=self._rtlsimpath +'/Z_' + rndpart +'.txt'
        self._rtlcmd=self.get_rtlcmd()

    def get_rtlcmd(self):
        #the could be gathered to rtl class in some way but they are now here for clarity
        submission = ' bsub -q normal '  
        rtllibcmd =  'vlib ' +  self._workpath + ' && sleep 2'
        rtllibmapcmd = 'vmap work ' + self._workpath

        if (self.model is 'vhdl'):
            rtlcompcmd = ( 'vcom ' + self._rtlsrcpath + '/' + self._name + '.vhd '
                          + self._rtlsrcpath + '/tb_'+ self._name+ '.vhd' )
            rtlsimcmd =  ( 'vsim -64 -batch -t 1ps -g g_infile=' + 
                           self._infile + ' -g g_outfile=' + self._outfile 
                           + ' work.tb_' + self._name + ' -do "run -all; quit -f;"')
            rtlcmd =  submission + rtllibcmd  +  ' && ' + rtllibmapcmd + ' && ' + rtlcompcmd +  ' && ' + rtlsimcmd

        elif (self.model is 'sv'):
            rtlcompcmd = ( 'vlog -work work ' + self._rtlsrcpath + '/' + self._name + '.sv '
                           + self._rtlsrcpath + '/tb_' + self._name +'.sv')
            rtlsimcmd = ( 'vsim -64 -batch -t 1ps -voptargs=+acc -g g_infile=' + self._infile
                          + ' -g g_outfile=' + self._outfile + ' work.tb_' + self._name  + ' -do "run -all; quit;"')

            rtlcmd =  submission + rtllibcmd  +  ' && ' + rtllibmapcmd + ' && ' + rtlcompcmd +  ' && ' + rtlsimcmd

        else:
            rtlcmd=[]
        return rtlcmd

    def run(self,*arg):
        if len(arg)>0:
            par=True      #flag for parallel processing
            queue=arg[0]  #multiprocessing.Queue as the first argument
        else:
            par=False

        if self.model=='py':
            out=np.array(self.iptr_A.Value)
            if par:
                queue.put(out)
            self._Z.Value=out
        else: 
          try:
              os.remove(self._infile)
          except:
              pass
          fid=open(self._infile,'wb')
          np.savetxt(fid,np.transpose(self.iptr_A.Value),fmt='%.0f')
          #np.savetxt(fid,np.transpose(inp),fmt='%.0f')
          fid.close()
          while not os.path.isfile(self._infile):
              #print("Wait infile to appear")
              time.sleep(1)
          try:
              os.remove(self._outfile)
          except:
              pass
          print("Running external command \n", self._rtlcmd , "\n" )
          subprocess.call(shlex.split(self._rtlcmd));
          
          while not os.path.isfile(self._outfile):
              #print("Wait outfile to appear")
              time.sleep(1)
          fid=open(self._outfile,'r')
          #fid=open(self._infile,'r')
          #out = .np.loadtxt(fid)
          out = np.transpose(np.loadtxt(fid))
          fid.close()
          if par:
              queue.put(out)
          self._Z.Value=out
          os.remove(self._infile)
          os.remove(self._outfile)

    def generate_halfbands(self,**kwargs):
       bandwidth=kwargs.get('bandwidth',0.45)
       n=kwargs.get('n',np.array([6,8,40]))
       H=[]
       for i in range(3):
           h=halfband()
           h.halfband_Bandwidth=bandwidth/(2**(2-i))
           h.halfband_N=n[i]
           h.init()
           h.export_scala()
           H.append(h)
       return H[0].H, H[2].H, H[2].H

if __name__=="__main__":
    import matplotlib.pyplot as plt
    d=f2_decimator()
    hb1, hb2, hb3 =d.generate_halfbands(**{'n':np.array([6,8,40]), 'bandwidth':0.45})
    impulse=np.r_['0', hb1, np.zeros((1024-hb1.shape[0],1))]
    impulse=np.r_['1', impulse, np.r_['0',hb2, np.zeros((1024-hb2.shape[0],1))]]
    impulse=np.r_['1', impulse, np.r_['0',hb3, np.zeros((1024-hb3.shape[0],1))]]
    total=np.convolve(hb3[0::2,0],hb2[:,0])
    total=np.convolve(total[0::2,],hb1[:,0]).reshape(-1,1)
    totimp=np.r_['0',total, np.zeros((1024-total.shape[0],1))]
    
    w=np.arange(1024)/1024
    spe1=np.fft.fft(impulse,axis=0)
    f=plt.figure(1)
    plt.plot(w,20*np.log10(np.abs(spe1)/np.amax(np.abs(spe1))))
    plt.ylim((-80,3))
    plt.grid(True)
    f.show()

    nbits=16
    spe3=np.fft.fft(np.round(impulse*(2**(nbits-1)-1)),axis=0)
    g=plt.figure(2)
    plt.plot(w,20*np.log10(np.abs(spe3)/np.amax(np.abs(spe3))))
    plt.ylim((-80,3))
    plt.grid(True)
    g.show()
    filts=[hb1, hb2, hb3]
    for i in range(len(filts)):
        tapfile=os.path.dirname(os.path.realpath(__file__)) +"/hb"+str(i+1)+".txt"
        fid=open(tapfile,'w')
        msg="val hb" +str(i+1)+"=Seq("
        fid.write(msg)
        lines=filts[i].shape[0]
        for k in range(lines-1):
            fid.write("%0.32f,\n" %(filts[i][k]))
        fid.write("%0.32f)\n" %(filts[i][lines-1]))


        #np.savetxt(fid,filts[i],fmt='%.32f')
        fid.close()
    #Required to keep the figures open
    input()



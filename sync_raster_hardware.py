'''
Created on Feb 4, 2015

@author: Hao Wu
Rewritten 2016-07-11 ESB
Rewritten 2017-01-27 ESB

'''
from qtpy import QtCore
from ScopeFoundry import HardwareComponent
from ScopeFoundry.helper_funcs import str2bool

try:
    from ScopeFoundryHW.ni_daq import NI_SyncTaskSet
except Exception as err:
    print("could not load modules needed for SyncRasterDAQ:", err)

import numpy as np

class SyncRasterDAQ(HardwareComponent):
    
    name = 'sync_raster_daq'
    
    # signal emitted when channels changed (adc, dac, ctr, enabled/disabled, name changes)
    channels_changed = QtCore.Signal() 

    
    def setup(self):
        self.display_update_period = 0.050 #seconds

        # Create logged quantities, set limits and defaults
        
        # adc rate
        self.settings.New("adc_rate", dtype=float, 
                            ro=False, 
                            initial=2e6, 
                            vmin=1, 
                            vmax=2e6,
                            unit='Hz',
                            si=True)
        
        # dac rate
        self.settings.New("dac_rate", dtype=float, 
                            ro=True, 
                            initial=5e5, 
                            vmin=1, 
                            vmax=2e6,
                            unit='Hz', si=True)
        
        # Ain_per_Aout Sample ratio
        # adc_oversample
        self.settings.New("adc_oversample", dtype=int, 
                            initial=1, 
                            vmin=1, vmax=1e10,
                            unit='x')
        
        
        self.continuous = self.settings.New('continuous', dtype=bool, initial=True)
        
        
        self.settings.New('adc_device', dtype=str, initial='Dev1')
        self.settings.New('dac_device', dtype=str, initial='Dev1')
        self.settings.New('ctr_device', dtype=str, initial='Dev1')

        self.settings.New('adc_channels', dtype=str, array=True, initial=['ai0', 'ai1'])
        self.settings.New('dac_channels', dtype=str, array=True, initial=['ao0', 'ao1'])
        self.settings.New('ctr_channels', dtype=str, array=True, initial=['ctr0', 'ctr1'])
        
        self.settings.New('adc_chans_enable', dtype=bool, array=True, initial=[1,1])
        self.settings.New('dac_chans_enable', dtype=bool, array=True, initial=[1,1])
        self.settings.New('ctr_chans_enable', dtype=bool, array=True, initial=[1,1])

        self.settings.New('adc_chan_names', dtype=str, array=True, initial=['ai0', 'ai1'])
        self.settings.New('dac_chan_names', dtype=str, array=True, initial=['ao0', 'ao1'])
        self.settings.New('ctr_chan_names', dtype=str, array=True, initial=['ctr0', 'ctr1'])
        
        self.settings.New('ctr_chan_terms', dtype=str, array=True, initial=['PFI0', 'PFI12'])
        
        self.settings.New('trig_output_term', dtype=str, array=False, initial='PXI_Trig0')
                 
        
        self.ext_clock_enable = self.add_logged_quantity("ext_clock_enable", dtype=bool, initial=False)
        self.ext_clock_source = self.add_logged_quantity("ext_clock_source", dtype=str, initial="/X-6368/PFI0")
        
        #parameters that cannot change during while connected
        self.lq_lock_on_connect = self.channel_lq_names =\
                                   ['adc_device', 'adc_channels', 'adc_chans_enable', 'adc_chan_names',
                                   'dac_device', 'dac_channels', 'dac_chans_enable', 'dac_chan_names',
                                   'ctr_device', 'ctr_channels', 'ctr_chans_enable', 'ctr_chan_names',
                                   'ctr_chan_terms', 'trig_output_term',
                                   ]
       
        # send channels_changed signal on change of these lq's
        for lq_name in self.channel_lq_names:
            self.settings.get_lq(lq_name).add_listener(lambda: self.channels_changed.emit())
       
        self.settings.adc_rate.add_listener(self.compute_dac_rate)
        self.settings.adc_oversample.add_listener(self.compute_dac_rate)
        
        
        
    def connect(self):        
        if self.debug_mode.val: self.log.debug( "connecting to {}".format(self.name))
        
        #self.remcon=self.app.hardware['sem_remcon']        
                
        # lock logged quantities during connection
        for lqname in self.lq_lock_on_connect:
            lq = self.settings.get_lq(lqname)
            lq.change_readonly(True)

        
        ## set up inputs to NI_SyncTaskSet
        if self.settings['ext_clock_enable']:
            clock_source = self.settings['ext_clock_source']
        else:
            clock_source = "" 
        

        # Select active channels for ADC, DAC, and counters                
        self.active_adc_chans = []
        self.active_adc_chan_names = []
        for i, chan in enumerate(self.settings['adc_channels']):
            if self.settings['adc_chans_enable'][i]:
                self.active_adc_chans.append( self.settings['adc_device'] + "/" + chan)
                self.active_adc_chan_names.append(self.settings['adc_chan_names'][i])
        adc_chan_str = ",".join( self.active_adc_chans )

        self.active_dac_chans = []
        self.active_dac_chan_names = []
        for i, chan in enumerate(self.settings['dac_channels']):
            if self.settings['dac_chans_enable'][i]:
                self.active_dac_chans.append( self.settings['dac_device'] + "/" + chan)
                self.active_dac_chan_names.append(self.settings['dac_chan_names'][i])
        dac_chan_str = ",".join( self.active_dac_chans )     
        
        self.active_ctr_chans = []
        self.active_ctr_terms = []
        self.active_ctr_chan_names = []

        for i, chan in enumerate(self.settings['ctr_channels']):
            if self.settings['ctr_chans_enable'][i]:
                self.active_ctr_chans.append(self.settings['ctr_device'] + '/' + chan)
                self.active_ctr_terms.append(self.settings['ctr_chan_terms'][i])
                self.active_ctr_chan_names.append(self.settings['ctr_chan_names'][i])
        
        
        ## create Sync Task set
        self.sync_analog_io = NI_SyncTaskSet(
                                   out_chan  = dac_chan_str,
                                   in_chan   = adc_chan_str,
                                   ctr_chans = self.active_ctr_chans,
                                   ctr_terms = self.active_ctr_terms,
                                   clock_source = clock_source,
                                   trigger_output_term = self.settings['trig_output_term'],
                                   )
        
        #from sample per point and sample rate, calculate the output(scan rate)
        #self.dac_rate.update_value(self.adc_rate.val/self.settings.adc_oversample.val)
        
        self.adc_chan_count  = self.sync_analog_io.get_adc_chan_count()
        assert self.adc_chan_count == len(self.active_adc_chans)
        
        assert self.num_ctrs == len(self.active_ctr_chans)
        
        
    def disconnect(self):
        
        for lqname in self.lq_lock_on_connect:
            lq = self.settings.get_lq(lqname)
            lq.change_readonly(False)

        #disconnect logged quantities from hardware
        self.settings.disconnect_all_from_hardware()

        #disconnect hardware
        if hasattr(self, "sync_analog_io"):
            self.sync_analog_io.stop()
            self.sync_analog_io.close()
        
            # clean up hardware object
            del self.sync_analog_io


    @property
    def num_ctrs(self):
        return self.sync_analog_io.num_ctrs
    
    def compute_dac_rate(self):
        self.settings['dac_rate'] = self.settings['adc_rate']/self.settings['adc_oversample']
    
    def setup_io_with_data(self, X, Y):
        """
        Set up sync task with X and Y arrays sent to the analog output channels
        Compute output rate based on settings 
        """
        assert len(X) == len(Y)        
        self.num_pixels = len(X)
        self.num_samples = int(self.num_pixels * self.settings['adc_oversample'])
        self.pixel_time = self.settings['adc_oversample'] / self.settings['adc_rate']        
        self.timeout = 1.5 * self.pixel_time * self.num_pixels             
        self.compute_dac_rate()
        
        
        self.sync_analog_io.setup(rate_out = self.settings['dac_rate'],
                                      count_out = self.num_pixels, 
                                      rate_in = self.settings['adc_rate'],
                                      count_in = self.num_samples, 
                                      is_finite=(not self.settings['continuous']))
        
        self.XY = self.interleave_xy_arrays(X, Y)                
        self.sync_analog_io.write_output_data_to_buffer(self.XY)
        
    def update_output_data(self, new_XY, timeout=0):
        self.current_XY = new_XY
        self.sync_analog_io.write_output_data_to_buffer(new_XY, timeout=timeout)
    
    def interleave_xy_arrays(self, X, Y):
        """take 1D X and Y arrays to create a flat interleaved XY array
        of the form [x0, y0, x1, y1, .... xN, yN]
        """
        assert len(X) == len(Y)
        N = len(X)
        XY = np.zeros(2*N, dtype=float)
        XY[0::2] = X
        XY[1::2] = Y
        return XY       
          

    def read_ai_chan_pixels(self, n_pixels):
        # Grabs n_pixels worth of multi-channel, multi-sample 
        # data shaped as (n_pixels, n_chan, n_samp)
        # TODO: check if n_pixels worth of data are actually returned
        n_samples = int(n_pixels * self.settings['adc_oversample'])
        buf = self.sync_analog_io.read_adc_buffer(count = n_samples, timeout=self.timeout)

        buf_reshaped = buf.reshape(n_pixels, self.settings['adc_oversample'], self.adc_chan_count).swapaxes(1,2)

        return buf_reshaped
    
    def read_counter_buffer(self, ctr_i, count=0):
        return self.sync_analog_io.read_ctr_buffer_diff(
            ctr_i, count, self.timeout)

    def start(self):
        # TODO disable LQ's that can't be changed during task run
        self.sync_analog_io.start()
        
    def stop(self):
        # TODO re-enable LQ's that can't be changed during task run
        self.sync_analog_io.stop()
        
        
    def set_n_pixel_callback_adc(self, n_pixels, adc_cb_func):
        """
        Setup callback functions for EveryNSamplesEvent
        *cb_func* will be called 
        after every *n_pixels* are acquired. 
        """
        n_samples = n_pixels*self.settings['adc_oversample']
        self.sync_analog_io.adc.set_n_sample_callback(n_samples, adc_cb_func)

    def set_n_pixel_callback_dac(self, n_pixels, dac_cb_func):
        """
        Setup callback functions for EveryNSamplesEvent
        *cb_func* will be called 
        after every *n_pixels* are acquired. 
        """
        self.sync_analog_io.dac.set_n_sample_callback(n_pixels, dac_cb_func)
        
        
    
    def set_ctr_n_pixel_callback(self, ctr_i, n_pixels, cb_func):
        """
        Setup callback functions for EveryNSamplesEvent
        *cb_func* will be called 
        after every *n_pixels* are acquired. 
        """
        n_samples = n_pixels#*self.settings.adc_oversample.val
        self.sync_analog_io.ctrs[ctr_i].set_n_sample_callback(n_samples, cb_func)
        
        

    #def read_counters(self):
    
    # replaced by callback-based measurement 2/9/17
    # keep as as reference for single frame measurement
    """
    def single_scan_data_block(self):
        self.ai_data = self.read_ai_chans()
            #handle oversampled ADC data
        self.ai_data =\
            self.ai_data.reshape(-1,self.settings.adc_oversample.val,self.adc_chan_count)
        self.ai_data = self.ai_data.mean(axis=1)
        
        self.sync_analog_io.stop()
        #self.scanDAQ.sync_analog_io.close()
        return self.ai_data

    # replaced by callback-based measurement 2/9/17
    # keep as as reference for single frame measurement
    def single_scan_regular(self, X_pos, Y_pos):
        #connect to SEM scanner module, which calculates the voltage output,
        #create detector channels and creates the scanning task

        self.setup_io_with_data(X_pos, Y_pos)
        self.sync_analog_io.start()            
        
        self.ai_data = self.read_ai_chans()
            #handle oversampled ADC data
        self.ai_data =\
            self.ai_data.reshape(-1,self.settings.adc_oversample.val,self.sync_analog_io.get_adc_chan_count())
        self.ai_data = self.ai_data.mean(axis=1)
        
        self.sync_analog_io.stop()
        #self.scanDAQ.sync_analog_io.close()
        return self.ai_data
    
    def read_ai_buffer(self):
        # interleaved buffer
        return self.sync_analog_io.read_adc_buffer(timeout=self.timeout)
    
    def read_ai_chans(self):
        return self.sync_analog_io.read_adc_buffer_reshaped(timeout=self.timeout)

    """
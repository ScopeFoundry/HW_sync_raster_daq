'''

Hao Wu  Feb 4, 2015
ESB 2016-07-19
ESB 2017-02-17

'''

from ScopeFoundry.scanning import BaseRaster2DScan
from ScopeFoundry import h5_io
import numpy as np
import time
from ScopeFoundry.helper_funcs import load_qt_ui_file, sibling_path,\
    replace_spinbox_in_layout
from ctypes import c_int32, c_uint32, c_uint64, byref
import PyDAQmx as mx
from .drift_correction import register_translation_hybrid
from ScopeFoundry.logged_quantity import LoggedQuantity, LQCollection
from collections import OrderedDict, namedtuple

AvailChan = namedtuple('AvailChan', ['type_', 'index', 'phys_chan', 'chan_name', 'term'])


class SyncRasterScan(BaseRaster2DScan):

    name = "sync_raster_scan"
    
    def setup(self):
        self.h_unit = self.v_unit = "V"
        self.h_limits = self.v_limits = (-10,10)
        
        BaseRaster2DScan.setup(self)
        self.Nh.update_value(1000)
        self.Nv.update_value(1000)
                
        self.display_update_period = 0.050 #seconds
        
        self.settings.New('adc_rate', dtype=float,initial = 500e3, unit='Hz')
        self.settings.New("adc_oversample", dtype=int, 
                            initial=1, 
                            vmin=1, vmax=1e10,
                            unit='x')
        self.disp_chan_choices = ['adc0', 'adc1', 'ctr0', 'ctr1'] 
        self.settings.New("display_chan", dtype=str, initial='adc0', choices=tuple(self.disp_chan_choices))
        self.settings.New("correct_drift", dtype=bool, initial=False)
        self.settings.New("correct_chan", dtype=int, initial=1, vmin=0, vmax=1)        
        self.settings.New("correlation_exp", dtype=float, initial=0.3, vmin=0., vmax=1.)
        self.settings.New("proportional_gain", dtype=float, initial=0.3, vmin=0.)
        # For now, these are just indicators
        self.settings.New('dac_offset_x', dtype=float, ro=True, unit='V', vmin = -10., vmax = 10.)
        self.settings.New('dac_offset_y', dtype=float, ro=True, unit='V', vmin = -10., vmax = 10.)
        
        self.scanDAQ = self.app.hardware['sync_raster_daq']        
        self.scan_on=False
        self.read_from_hardware=True #Disabling allows fast repeated "single scans"
        
        self.details_ui = load_qt_ui_file(sibling_path(__file__, 'sync_raster_details.ui'))
        self.ui.details_groupBox.layout().addWidget(self.details_ui) # comment out?
        
        # self.ui.setWindowTitle('sync_raster_scan') #restore?
        
        self.settings.n_frames.connect_to_widget(self.details_ui.n_frames_doubleSpinBox)
        self.settings.adc_oversample.connect_to_widget(self.details_ui.adc_oversample_doubleSpinBox)
        #self.settings.adc_rate.connect_to_widget(self.details_ui.adc_rate_doubleSpinBox)
        self.details_ui.adc_rate_pgSpinBox = \
            replace_spinbox_in_layout(self.details_ui.adc_rate_doubleSpinBox)
        self.settings.adc_rate.connect_to_widget(
            self.details_ui.adc_rate_pgSpinBox)
        self.settings.display_chan.connect_to_widget(self.details_ui.display_chan_comboBox)
        
        self.details_ui.pixel_time_pgSpinBox = \
            replace_spinbox_in_layout(self.details_ui.pixel_time_doubleSpinBox)
        self.settings.pixel_time.connect_to_widget(
            self.details_ui.pixel_time_pgSpinBox)
        
        self.details_ui.line_time_pgSpinBox = \
            replace_spinbox_in_layout(self.details_ui.line_time_doubleSpinBox)
        self.settings.line_time.connect_to_widget(
            self.details_ui.line_time_pgSpinBox)
        
        self.details_ui.frame_time_pgSpinBox = \
            replace_spinbox_in_layout(self.details_ui.frame_time_doubleSpinBox)
        self.settings.frame_time.connect_to_widget(
            self.details_ui.frame_time_pgSpinBox)

        
        self.scanDAQ.settings.dac_rate.add_listener(self.compute_times)
        self.settings.Nh.add_listener(self.compute_times)
        self.settings.Nv.add_listener(self.compute_times)
        
        
        if hasattr(self.app,'sem_remcon'):#FIX re-implement later
            self.sem_remcon=self.app.sem_remcon
        
        S = self.settings
        self.settings.pixel_time.connect_lq_math([S.adc_rate,S.adc_oversample],
                                                 lambda rate, oversample: oversample/rate)
        
#     def dock_config(self):
#         
#         del self.ui.plot_groupBox
#         
#         
#         self.dockarea.addDock(name='SEM Sync Settings', position='left', widget=self.sem_controls)
# 
#         self.dockarea.addDock(name='Details', position='right', widget=self.details_ui)
# 
#         self.dockarea.addDock(name='Image', position='bottom', widget=self.graph_layout)
#         
# 
#     WIP

    def pre_run(self):
        self.scanDAQ = self.app.hardware['sync_raster_daq']        
        self.scanDAQ.settings['adc_rate'] = self.settings['adc_rate']
        self.scanDAQ.settings['adc_oversample'] = self.settings['adc_oversample']
        self.scanDAQ.compute_dac_rate()

    
    def run(self):
        
        self.frame_time_i = time.time()
        
        self.on_start_of_run()
        
        self.adc_poll_period = 0.050
        
        # if hardware is not connected, connect it
        #time.sleep(0.5)
        if not self.scanDAQ.settings['connected']:
            self.scanDAQ.settings['connected'] = True
            # we need to wait while the task is created before 
            # measurement thread continues
            time.sleep(0.1)
            
        

            # READ FROM HARDWARE BEFORE SCANNING -- Drift correction depends on accurate numbers
            # also disable beam blank, enable ext scan
        if self.read_from_hardware:
            self.app.hardware['sem_remcon'].read_from_hardware()
        self.app.hardware['sem_remcon'].settings['external_scan'] = 1
        self.app.hardware['sem_remcon'].settings['beam_blanking'] = 0
       
            # Compute data arrays (scanDAQ.XY)        
        self.compute_scan_arrays()        
        self.initial_scan_setup_plotting = True        
        self.display_image_map = np.zeros(self.scan_shape, dtype=float)
    
            # Initialize quantities for drift correction
        self.win = np.outer(np.hanning(self.settings['Nv']),np.hanning(self.settings['Nh']))
                                
        self.in_dac_callback = False #flag to prevent reentrant execution
        
        try:
            if self.settings['save_h5']:
                self.h5_file = h5_io.h5_base_file(self.app, measurement=self)
                self.h5_filename = self.h5_file.filename
                self.h5_m = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5_file)
                self.display_update_period = 0.05
            else:
                self.display_update_period = 0.01

                ##### Start indexing            
            self.total_pixel_index = 0 # contains index of next adc pixel to be moved from queue into h5 file
            self.pixel_index = 0 # contains index of next adc pixel to be moved from queue into adc_pixels (within frame display)
            self.current_scan_index = self.scan_index_array[0]
            self.task_done = False
                       
                ##### load initial XY positions in to DAC
            self.dac_taskhandle = self.scanDAQ.sync_analog_io.dac.task.taskHandle
                #allow regen lets DAQ loop over scan buffer, do before task starts
                #callbacks optionally dynamically update buffer for drift
            mx.DAQmxSetWriteRegenMode(self.dac_taskhandle, mx.DAQmx_Val_AllowRegen)

                #FIX kludge so python image matches SEM image
            self.scanDAQ.setup_io_with_data(self.scan_h_positions, -1*self.scan_v_positions)
            
            self.orig_XY = self.scanDAQ.XY.copy() # copy of the original raster            
            self.current_frame_XY = self.orig_XY.copy()
            
            #callback outputs this buffer, which may be adjusted by drift correction
            self.next_frame_XY = self.orig_XY.copy()
            self.next_frame_XY_ready = False

            

            #c-type variables used by DAQmx status functions
            self.space_available = c_uint32()
            self.write_pos = c_uint64()
            self.samples_generated = c_uint64() 
            self.dac_callback_elapsed = time.time()
            
            mx.DAQmxGetWriteSpaceAvail(self.dac_taskhandle, byref(self.space_available))
            self.scanDAQ.buffer_size = self.space_available.value
            self.log.info("Initial DAQmxGetWriteSpaceAvail {}".format( self.space_available.value))
            

            ###### compute pixel acquisition block size 
            # need at least one, and needs to an integer divisor of Npixels            
                #ADC input callback block size
            num_pixels_per_block = max(1, int(np.ceil(self.adc_poll_period / self.scanDAQ.pixel_time)))
                #force block to be an integer number of horizontal scan lines
            if num_pixels_per_block > self.Nh.val:
                num_pixels_per_block = self.Nh.val*np.ceil( num_pixels_per_block / self.Nh.val )                
            num_blocks = int(max(1, np.floor(self.Npixels / num_pixels_per_block)))
            
            
            #force/calc integer number of adc blocks per image
            while self.Npixels % num_blocks != 0:
                num_blocks -= 1
            self.num_pixels_per_block = num_pixels_per_block = int(self.Npixels / num_blocks)
            self.log.info("num_pixels_per_block {}".format( num_pixels_per_block))

                # minimum DAC callback half an image since initial buffer is one image
            self.num_pixels_per_dac_block = int(self.Npixels/2)
            self.time_per_dac_block = self.num_pixels_per_dac_block * self.scanDAQ.pixel_time            
            assert self.num_pixels_per_dac_block*2 == self.Npixels #Npixels needs to be even...            
            self.log.info("num_pixels_per_dac_block {}".format( self.num_pixels_per_dac_block))        
            
            
            ##### Data array
            # ADC
            self.adc_pixels = np.zeros((self.Npixels, self.scanDAQ.adc_chan_count), dtype=float)
            self.new_adc_data_queue = [] # will contain numpy arrays (data blocks) from adc to be processed
            self.adc_map = np.zeros(self.scan_shape + (self.scanDAQ.adc_chan_count,), dtype=float)
            
            adc_chunk_size = (1,1, max(1,num_pixels_per_block/self.Nh.val), self.Nh.val ,self.scanDAQ.adc_chan_count )
            self.log.info('adc_chunk_size {}'.format(adc_chunk_size))
            self.adc_map_h5 = self.create_h5_framed_dataset('adc_map', self.adc_map, chunks=adc_chunk_size, compression=None)
                    
            # Ctr
            # ctr_pixel_index contains index of next pixel to be processed, 
            # need one per ctr since ctrs are independent tasks
            self.ctr_pixel_index = np.zeros(self.scanDAQ.num_ctrs, dtype=int)
            self.ctr_total_pixel_index = np.zeros(self.scanDAQ.num_ctrs, dtype=int)
            self.ctr_pixels = np.zeros((self.Npixels, self.scanDAQ.num_ctrs), dtype=int)
            self.new_ctr_data_queue = [] # list will contain tuples (ctr_number, data_block) to be processed
            self.ctr_map = np.zeros(self.scan_shape + (self.scanDAQ.num_ctrs,), dtype=int)
            self.ctr_map_Hz = np.zeros(self.ctr_map.shape, dtype=float)
            ctr_chunk_size = (1,1, max(1,num_pixels_per_block/self.Nh.val), self.Nh.val, self.scanDAQ.num_ctrs)
            self.log.info('ctr_chunk_size {}'.format(ctr_chunk_size))
            self.ctr_map_h5 = self.create_h5_framed_dataset('ctr_map', self.ctr_map, chunks=ctr_chunk_size, compression=None)
            
            # Drift vector stored to h5
            self.dac_offsets = [np.zeros(2, dtype=float)] # list of pairs of (dx,dy) in volts
            self.dac_offset_h5 = self.create_h5_framed_dataset('dac_offset', self.dac_offsets[0])
            
            
#             self.h5_m['drift_vec'] = np.zeros((self.npoints))

            
                        
            ##### register callbacks
            if hasattr(self.scanDAQ.sync_analog_io, 'sync_raster_scan_callbacks'):
                print("callbacks already defined")
            else:
                self.scanDAQ.set_n_pixel_callback_adc(
                    num_pixels_per_block, 
                    self.every_n_callback_func_adc)
                
                self.scanDAQ.set_n_pixel_callback_dac(
                    self.num_pixels_per_dac_block,
                    self.every_n_callback_func_dac)
                
                self.scanDAQ.sync_analog_io.adc.set_done_callback(
                    self.done_callback_func_adc )
                
                self.dac_i = 0
                
                for ctr_i in range(self.scanDAQ.num_ctrs):
                    self.scanDAQ.set_ctr_n_pixel_callback( ctr_i,
                            num_pixels_per_block, lambda i=ctr_i: self.every_n_callback_func_ctr(i))

                self.scanDAQ.sync_analog_io.sync_raster_scan_callbacks = True
            
            self.pre_scan_setup()

            #### Start scan daq 
            self.scanDAQ.start()
            
            #### Wait until done, while processing data queues
            while not self.task_done and not self.interrupt_measurement_called:
                self.handle_new_data()
                time.sleep(self.adc_poll_period)
                
            # FIX handle serpentine scans
            #self.display_image_map[self.scan_index_array] = self.ai_data[0,:]
            # TODO save data
            

        finally:
            # When done, stop tasks
            if self.settings['save_h5']:
                self.log.info('data saved to {}'.format(self.h5_file.filename))
                self.h5_file.close()            
            self.scanDAQ.stop()
            
            ##### TODO unregister callbacks
            self.scanDAQ.settings['connected']=False
            
            
            # TODO disconnect callback
            #self.scanDAQ.
            #print("Npixels", self.Npixels, 'block size', self.num_pixels_per_block, 'num_blocks', num_blocks)
            #print("pixels remaining:", self.pixels_remaining)
            #print("blocks_per_sec",1.0/ (self.scanDAQ.pixel_time*num_pixels_per_block))
            #print("frames_per_sec",1.0/ (self.scanDAQ.pixel_time*self.Npixels))
            
            # Update the H, V raster values to the dac offsets so scan window displacement is visible
            # Also allows next scan to start in the same location
            if self.settings['correct_drift']:
                self.settings['h0'] += self.dac_offsets[-1][0]
                self.settings['h1'] += self.dac_offsets[-1][0]
                self.settings['v0'] -= self.dac_offsets[-1][1]
                self.settings['v1'] -= self.dac_offsets[-1][1]
            
            self.post_scan_cleanup()
            
            print(self.name, "done")

        
    
    def update_display(self):
        self.get_display_pixels()
        x = self.scan_index_array.T
        self.display_image_map[x[0], x[1], x[2]] = self.display_pixels

        kk,jj, ii = self.scan_index_array[self.pixel_index]
        #self.current_stage_pos_arrow.setPos(self.h_array[ii], self.v_array[jj])
        self.current_stage_pos_arrow.setVisible(False)
        t0 = time.time()
        BaseRaster2DScan.update_display(self)
        #print("sync_raster_scan timing {}".format(time.time()-t0))
    
    ##### Callback functions
    def every_n_callback_func_adc(self):
        new_adc_data = self.scanDAQ.read_ai_chan_pixels(
            self.num_pixels_per_block)
        self.new_adc_data_queue.append(new_adc_data)
        #self.on_new_adc_data(new_data)
        num_new_pixels = new_adc_data.shape[0]
        pixel_index = self.pixel_index + num_new_pixels
        total_pixel_index =  self.total_pixel_index + num_new_pixels
        pixel_index %= self.Npixels
        if pixel_index == 0:
            frame_num = (total_pixel_index // self.Npixels) - 1
            self.on_new_frame(frame_num)
        
        return 0
    
    def every_n_callback_func_dac(self):
        '''
        DAQmx output callback notes
            DAQmxGetWriteTotalSampPerChanGenerated
                is actual samples generated, may be some latency with DAQmx to hardware blocks...
            DAQmxGetWriteCurrWritePos
                flat pointer to output buffer write position,
                ie does not wrap with finite buffer size.
            DAQmxGetWriteSpaceAvail
                space between WritePos and GenPointer
        
        '''
        if self.in_dac_callback:
            print("this should not be! Re-entry into callback ")
        self.in_dac_callback = True
        self.dac_callback_elapsed = time.time()
                
        self.dac_percent_frame = self.dac_i*100.0/self.Npixels  
        try:
            #data = c_int32()
            #mx.DAQmxGetWriteOffset(self.dac_taskhandle, byref(data))
            #print("DAQmxGetWriteOffset", data.value)
            #mx.DAQmxGetWriteRelativeTo(self.dac_taskhandle, byref(data) )
            #print("DAQmxGetWriteRelativeTo", data.value)

            #sometimes these callus ? 10 ms...
            mx.DAQmxGetWriteCurrWritePos(self.dac_taskhandle, byref(self.write_pos))
            mx.DAQmxGetWriteSpaceAvail(self.dac_taskhandle, self.space_available)
            mx.DAQmxGetWriteTotalSampPerChanGenerated(
                self.dac_taskhandle, byref(self.samples_generated))
            
            # update DAC output array 
            ii = self.dac_i
            
            if self.next_frame_XY_ready and self.dac_i == 0:
                self.current_frame_XY = self.next_frame_XY.copy()
                self.next_frame_XY_ready = False
            
            # note: convert pixel index ii to data output index (2*ii) for two channels (x,y)
            self.scanDAQ.update_output_data(
                self.current_frame_XY[2*ii:2*ii+2*self.num_pixels_per_dac_block],
                timeout=0.0 )
            self.dac_i += self.num_pixels_per_dac_block
            self.dac_i %= self.Npixels            

        finally:
            self.in_dac_callback = False
            self.dac_callback_elapsed = time.time() - self.dac_callback_elapsed
#             print("DAQ elapsed time {:.3g} ms {}% frame write pos {:d} space {:d} samples {:d}"\
#                   .format(self.dac_callback_elapsed*1e3, self.dac_percent_frame,\
#                           self.write_pos.value,\
#                           self.space_available.value,\
#                           self.samples_generated.value))

        return 0
    
    def every_n_callback_func_ctr(self, ctr_i):
        new_ctr_data = self.scanDAQ.read_counter_buffer(
            ctr_i, self.num_pixels_per_block)
        self.new_ctr_data_queue.append( (ctr_i, new_ctr_data))
        #print("every_n_callback_func_ctr {} {}".format(ctr_i, len(new_ctr_data)))
        return 0
            
    def done_callback_func_adc(self, status):
        self.task_done = True
        #print("done", status)
        return 0
    
    def handle_new_data(self):
        while len(self.new_adc_data_queue) > 0:
            # grab the next available data chunk
            #print('new_adc_data_queue' + "[===] "*len(self.new_adc_data_queue))
            new_data = self.new_adc_data_queue.pop(0)
            self.on_new_adc_data(new_data)
            if self.interrupt_measurement_called:
                break

        while len(self.new_ctr_data_queue) > 0:
            ctr_i, new_data = self.new_ctr_data_queue.pop(0)
            self.on_new_ctr_data(ctr_i, new_data)
            if self.interrupt_measurement_called:
                break

    
    def on_new_adc_data(self, new_data):
        self.set_progress(100*self.pixel_index / self.Npixels )
        #print('callback block', self.pixel_index, new_data.shape, 'remaining px', self.Npixels - self.pixel_index)
        ii = self.pixel_index
        dii = num_new_pixels = new_data.shape[0]
        # average over samples (takes oversampled adc data and
        # gives single pixel average for each channel)
        new_data = new_data.mean(axis=2)

        #stuff into pixel data array
        self.adc_pixels[ii:ii+dii , :] = new_data
                
        self.current_scan_index = self.scan_index_array[self.pixel_index]

        self.pixel_index += num_new_pixels
        self.total_pixel_index += num_new_pixels
        
        self.pixel_index %= self.Npixels
        
        
        # copy data to image shaped map
        x = self.scan_index_array[ii:ii+dii,:].T
        self.adc_map[x[0], x[1], x[2],:] = new_data

        # Frame complete
        #pixels_remaining = self.Npixels - self.pixel_index
        #print("adc pixels_remaining", self.pixel_index, pixels_remaining, self.Npixels, frame_num)
        if self.pixel_index == 0:
            frame_num = (self.total_pixel_index // self.Npixels) - 1
            # Copy data to H5 file, if a frame is complete
            if self.settings['save_h5']:
                #print("saving h5 adc", frame_num)
                self.extend_h5_framed_dataset(self.adc_map_h5, frame_num)
                self.adc_map_h5[frame_num, :,:,:,:] = self.adc_map
                self.h5_file.flush()
            
            self.on_end_frame(frame_num) # removed -1 !!!!
            
            # Stop scan if n_frames reached:
            if (not self.settings['continuous_scan']) \
                    and (frame_num >= self.settings['n_frames'] - 1) :
                self.task_done = True
            
    
    def on_new_frame(self, frame_i):
        pass
    
    def on_end_frame(self, frame_i):
        if hasattr(self, "frame_time_i"):
            print("sync_raster_scan frame_time", time.time() - self.frame_time_i)
        self.frame_time_i = time.time()
        
        if self.settings['correct_drift']:
            frame_num = frame_i
            print('frame_num',frame_num)
            
            if frame_num == 0:
                #Reference image
                self.ref_image = self.adc_map[0,:,:,self.settings['correct_chan']].copy()
                print('reference image stored')
                self.cumul_shift = [(0., 0.)] # Initialize cumulative shifts incurred over this scan
                print('Cumulative shift (x,y)', self.cumul_shift[-1])
            else:
                #Offset image
                self.offset_image = self.adc_map[0,:,:,self.settings['correct_chan']] #assumes no subframes
                print('map shape', self.adc_map.shape)
                print('current image stored')
                            
                # Shift determination
                shift, error, diffphase = register_translation_hybrid(self.ref_image*self.win, self.offset_image*self.win, 
                                                                           exponent = self.settings['correlation_exp'], upsample_factor = 100)
                print('Image shift [px]', shift)
                # Shift defined as [y, x] vector in direction of motion of view relative to sample
                # pos x shifts view to the right
                # pos y shifts view upwards
                
                # shift_factor converts pixel shift to voltage shift
                shift_factor_h = -1*(self.settings['h1']-self.settings['h0'])/self.settings['Nh'] # V/px
                shift_factor_v = (self.settings['v1']-self.settings['v0'])/self.settings['Nv'] # V/px
                
                # Proportional gain reduces correction in anticipation of overcorrection due to not applying
                # correction until an image later than prescribed
                P = self.settings['proportional_gain']
                # cumul_shift keeps track of cumulative 'measured' shifts
                self.cumul_shift += [(self.cumul_shift[-1][0] + shift_factor_h * shift[1], 
                                      self.cumul_shift[-1][1] + shift_factor_v * shift[0])]   # Volts
                print('Cumulative shift (x,y)', self.cumul_shift[-1])
                

                self.dac_offsets += [P * np.array(self.cumul_shift[-1])]
                print('DAC offset (x,y)', self.dac_offsets[-1])
                self.settings['dac_offset_x'] = self.dac_offsets[-1][0]
                self.settings['dac_offset_y'] = self.dac_offsets[-1][1]
                
                self.next_frame_XY[0::2] = self.orig_XY[0::2] + self.dac_offsets[-1][0]  # shift x raster
                self.next_frame_XY[1::2] = self.orig_XY[1::2] + self.dac_offsets[-1][1]  # shift y raster
                
                self.next_frame_XY_ready = True
                
                if self.settings['save_h5']:
                    self.extend_h5_framed_dataset(self.dac_offset_h5, frame_num)
                    # dac offset keeps track of 'applied' offsets, which is P * shifts
                    
#                     self.settings['dac_offset'] = self.dac_offset[-1]
                    self.dac_offset_h5[frame_i,:] = self.dac_offsets[-1]
                    self.h5_file.flush()
                
    def on_new_ctr_data(self, ctr_i, new_data):
        #print("on_new_ctr_data {} {}".format(ctr_i, new_data))
        ii = self.ctr_pixel_index[ctr_i]
        dii = num_new_pixels = new_data.shape[0]
        
        self.ctr_pixels[ii: ii+dii, ctr_i] = new_data
        
        self.ctr_pixel_index[ctr_i] += dii
        self.ctr_total_pixel_index[ctr_i] += dii
        self.ctr_pixel_index[ctr_i] %= self.Npixels
        
        # copy pixel 1 to pixel 0 to avoid large count number 
        # from free-running counter between measurements
        if ii == 0 and dii > 1:
            new_data[0] = new_data[1]
        
        # copy data to image shaped map
        x = self.scan_index_array[ii:ii+dii,:].T
        self.ctr_map[x[0], x[1], x[2], ctr_i] = new_data
        self.ctr_map_Hz[x[0], x[1], x[2], ctr_i] = new_data *1.0/ self.scanDAQ.pixel_time

        # Frame complete
        if self.ctr_pixel_index[ctr_i] == 0:
            frame_num = (self.ctr_total_pixel_index[ctr_i] // self.Npixels) - 1
            #print('ctr frame complete', frame_num)
            # Copy data to H5 file, if a frame is complete
            if self.settings['save_h5']:
                #print('save data ctr')
                self.extend_h5_framed_dataset(self.ctr_map_h5, frame_num)
                self.ctr_map_h5[frame_num,:,:,:,ctr_i] = self.ctr_map[:,:,:,ctr_i]
                self.h5_file.flush()
        

    def on_start_of_run(self):
        pass
    
    def pre_scan_setup(self):
        pass

    def post_scan_cleanup(self):
        pass
    
    def get_display_pixels(self):
        #DISPLAY_CHAN = 0
        #self.display_pixels = self.adc_pixels[:,DISPLAY_CHAN]
        #self.display_pixels[0] = 0
        
        chan_data = {
            'adc0': self.adc_pixels[:,0],
            'adc1': self.adc_pixels[:,1],
            'ctr0': self.ctr_pixels[:,0],
            'ctr1': self.ctr_pixels[:,1],
            }
        
        self.display_pixels = chan_data[self.settings['display_chan']]
        
        #self.display_pixels = self.ctr_pixels[:,DISPLAY_CHAN]
        
    def create_h5_framed_dataset(self, name, single_frame_map, **kwargs):
        """
        Create and return an empty HDF5 dataset in self.h5_m that can store
        multiple frames of single_frame_map.
        
        Must fill the dataset as frames roll in.
        
        creates reasonable defaults for compression and dtype, can be overriden 
        with**kwargs are sent directly to create_dataset
        """
        if self.settings['save_h5']:
            shape=(self.settings['n_frames'],) + single_frame_map.shape
            if self.settings['continuous_scan']:
                # allow for array to grow to store additional frames
                maxshape = (None,)+single_frame_map.shape 
            else:
                maxshape = shape
            print('maxshape', maxshape)
            default_kwargs = dict(
                name=name,
                shape=shape,
                dtype=single_frame_map.dtype,
                #chunks=(1,),
                chunks=(1,)+single_frame_map.shape,
                maxshape=maxshape,
                compression='gzip',
                #shuffle=True,
                )
            default_kwargs.update(kwargs)
            map_h5 =  self.h5_m.create_dataset(
                **default_kwargs
                )
            return map_h5
    
    def extend_h5_framed_dataset(self, map_h5, frame_num):
        """
        Adds additional frames to dataset map_h5, if frame_num 
        is too large. Adds n_frames worth of extra frames
        """
        if self.settings['continuous_scan']:
            current_num_frames, *frame_shape = map_h5.shape
            if frame_num >= current_num_frames:
                print ("extend_h5_framed_dataset", map_h5.name, map_h5.shape, frame_num)
                n_frames_extend = self.settings['n_frames']
                new_num_frames = n_frames_extend*(1 + frame_num//n_frames_extend)
                map_h5.resize((new_num_frames,) + tuple(frame_shape))
                return True
            else:
                return False
        else:
            return False
    
#     def compute_times(self):
#         #if hasattr(self, 'scanDAQ'):
#         dac_rate = self.settings['adc_rate']*self.settings['adc_oversample']
#         self.settings['pixel_time'] = 1.0/dac_rate
#         BaseRaster2DScan.compute_times(self)

    def update_available_channels(self):
        self.available_chan_dict = OrderedDict()
                
        for i, phys_chan in enumerate(self.scanDAQ.settings['adc_channels']):
            self.available_chan_dict[phys_chan] = AvailChan(
                # type, index, physical_chan, channel_name, terminal
                'ai', i, phys_chan, self.scanDAQ.settings['adc_chan_names'][i], phys_chan)
        for i, phys_chan in enumerate(self.scanDAQ.settings['ctr_channels']):
            self.available_chan_dict[phys_chan] = AvailChan(
                # type, index, physical_chan, channel_name, terminal
                'ctr', i, phys_chan, self.scanDAQ.settings['ctr_chan_names'][i], self.scanDAQ.settings['ctr_chan_terms'][i])


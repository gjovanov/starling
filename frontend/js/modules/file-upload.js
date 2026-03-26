/**
 * File upload module for handling drag-and-drop and click-to-upload functionality
 */

export class FileUploadHandler {
  /**
   * @param {Object} options
   * @param {HTMLElement} options.dropZone - The drop zone element
   * @param {HTMLInputElement} options.fileInput - The file input element
   * @param {Function} options.onUpload - Callback function when file is selected
   * @param {string[]} options.acceptedTypes - Array of accepted file extensions (e.g., ['.wav', '.mp3'])
   * @param {Function} options.onError - Optional error callback
   */
  constructor({ dropZone, fileInput, onUpload, acceptedTypes = ['.wav', '.mp3'], onError }) {
    this.dropZone = dropZone;
    this.fileInput = fileInput;
    this.onUpload = onUpload;
    this.acceptedTypes = acceptedTypes;
    this.onError = onError || ((msg) => alert(msg));

    if (!this.dropZone || !this.fileInput) {
      console.warn('[FileUpload] Missing elements:', { dropZone: !!dropZone, fileInput: !!fileInput });
      return;
    }

    this._setupEventListeners();
    console.log('[FileUpload] Initialized successfully');
  }

  _setupEventListeners() {
    // Click to open file dialog
    this.dropZone.addEventListener('click', (e) => {
      // Don't trigger if clicking on a child element that handles its own clicks
      if (e.target === this.fileInput) return;
      console.log('[FileUpload] Drop zone clicked');
      this.fileInput.click();
    });

    // Drag over - show visual feedback
    this.dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.dropZone.classList.add('dragover');
    });

    // Drag enter - additional visual feedback
    this.dropZone.addEventListener('dragenter', (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.dropZone.classList.add('dragover');
    });

    // Drag leave - remove visual feedback
    this.dropZone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      e.stopPropagation();
      // Only remove class if leaving the drop zone entirely
      if (!this.dropZone.contains(e.relatedTarget)) {
        this.dropZone.classList.remove('dragover');
      }
    });

    // Drop - handle the dropped file
    this.dropZone.addEventListener('drop', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.dropZone.classList.remove('dragover');

      const file = e.dataTransfer?.files[0];
      if (file) {
        console.log('[FileUpload] File dropped:', file.name);
        await this._handleFile(file);
      }
    });

    // File input change - handle selected file
    this.fileInput.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      console.log('[FileUpload] Change event, file:', file?.name);
      if (file) {
        console.log('[FileUpload] File selected:', file.name);
        try {
          await this._handleFile(file);
        } catch (err) {
          console.error('[FileUpload] Error in _handleFile:', err);
        }
      }
      // Reset input to allow selecting the same file again
      e.target.value = '';
    });
  }

  async _handleFile(file) {
    console.log('[FileUpload] _handleFile called:', file.name, file.size);

    // Validate file type
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    console.log('[FileUpload] File extension:', ext, 'Accepted:', this.acceptedTypes);

    if (!this.acceptedTypes.includes(ext)) {
      const msg = `Please upload a ${this.acceptedTypes.join(' or ').toUpperCase().replace(/\./g, '')} file`;
      console.warn('[FileUpload] Invalid file type:', file.name);
      this.onError(msg);
      return;
    }

    // Call the upload handler
    console.log('[FileUpload] Calling onUpload...');
    try {
      await this.onUpload(file);
      console.log('[FileUpload] onUpload completed successfully');
    } catch (error) {
      console.error('[FileUpload] Upload failed:', error);
      this.onError(`Upload failed: ${error.message}`);
    }
  }
}

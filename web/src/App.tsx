import { useRef, useState, useEffect } from "react";

// Configuration constants
const MAX_FILE_SIZE_MB = 250;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

type AppState = 'upload' | 'interim' | 'results' | 'error';

interface AnalysisResult {
  message: string;
  video_url?: string;
  analysis?: {
    score: number;
    maneuvers: Array<{name: string; start_time: number; end_time: number}>;
  };
}

const useIsMobile = () => {
  const [isMobile, setIsMobile] = useState(false);
  
  useEffect(() => {
    const isMobileUA = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    setIsMobile(isMobileUA);
  }, []);
  
  return isMobile;
};

const fetchWithTimeout = async (url: string, options: RequestInit, timeout = 3000): Promise<Response> => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('Server unavailable, please try again later.');
    }
    throw error;
  }
};

export default function App() {
  const [appState, setAppState] = useState<AppState>('upload');
  const [sseMessages, setSseMessages] = useState<string[]>([]);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [analysisStartTime, setAnalysisStartTime] = useState<Date | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isMobile = useIsMobile();

  // Timer effect for updating elapsed time
  useEffect(() => {
    if (!analysisStartTime) return;

    const interval = setInterval(() => {
      const now = new Date();
      const elapsed = Math.floor((now.getTime() - analysisStartTime.getTime()) / 100);
      setElapsedTime(elapsed);
    }, 100);

    return () => clearInterval(interval);
  }, [analysisStartTime]);

  // Helper function to format time
  const formatTime = (tenths: number) => {
    const totalSeconds = Math.floor(tenths / 10);
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    const tenthsOfSecond = tenths % 10;
    return `${mins}:${secs.toString().padStart(2, '0')}.${tenthsOfSecond}`;
  };

  // Helper function to format file size
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const validateFile = (file: File): string | null => {
    // Check file type
    if (!file.type.startsWith('video/')) {
      return 'Please select a video file (MP4, MOV, AVI)';
    }
    
    if (file.size > MAX_FILE_SIZE_BYTES) {
      return `File size must be less than ${MAX_FILE_SIZE_MB}MB`;
    }
    
    return null;
  };

  const uploadVideo = async (file: File) => {
    setIsUploading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetchWithTimeout(`${import.meta.env.VITE_API_BASE_URL}${import.meta.env.VITE_API_ENDPOINT}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        if (response.status === 413) {
          throw new Error('File too large. Please try a smaller video file.');
        } else if (response.status === 415) {
          throw new Error('Unsupported file type. Please upload a video file.');
        } else if (response.status >= 500) {
          throw new Error('Server error. Please try again later.');
        } else {
          throw new Error(`Upload failed (${response.status}). Please try again.`);
        }
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response from server. Please try again.');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.status === 'interim') {
                setSseMessages(prev => [...prev, data.message]);
              } else if (data.status === 'success') {
                setAnalysisResult({
                  message: data.message,
                  video_url: data.video_url,
                  analysis: data.analysis
                });
                setAppState('results');
                setIsUploading(false);
                return;
              } else if (data.status === 'server_error') {
                setError('Something went wrong. Please try again.');
                setAppState('error');
                setIsUploading(false);
                return;
              } else if (data.status === 'user_error') {
                setError(data.message);
                setAppState('error');
                setIsUploading(false);
                return;
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
              // Continue processing other messages
            }
          }
        }
      }
      
      // If we get here without success, something went wrong
      throw new Error('Analysis incomplete. Please try again.');
      
    } catch (error) {
      console.error('Upload error:', error);
      setError(error instanceof Error ? error.message : 'An unexpected error occurred. Please try again.');
      setAppState('error');
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileSelect = async (file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      setAppState('error');
      return;
    }
    
    setAppState('interim');
    setSseMessages([]);
    setAnalysisResult(null);
    setError(null);
    setUploadedFile(file);
    setAnalysisStartTime(new Date());
    
    // Start upload process
    await uploadVideo(file);
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingFiles(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.types.includes('Files')) {
      setIsDraggingFiles(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    // Only reset if we're leaving the drop zone entirely (not just moving over a child)
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDraggingFiles(false);
    }
  };

  const handleUploadAnother = () => {
    setAppState('upload');
    setSseMessages([]);
    setAnalysisResult(null);
    setError(null);
    setUploadedFile(null);
    setAnalysisStartTime(null);
    setElapsedTime(0);
  };

  // Render different states
  if (appState === 'interim') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-200">
        <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md flex flex-col items-center">
          <h1 className="text-2xl font-bold text-gray-800">Analyzing Video...</h1>
          
          <div className="text-gray-600 text-center">
            <br />
            <div className="space-y-1">
              {/* Show "Starting analysis..." as in-progress until first SSE message */}
              {sseMessages.length === 0 ? (
                <div className="flex items-center justify-center">
                  <div className="w-4 h-4 mr-2 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></div>
                  <span className="font-medium text-gray-800">Starting analysis...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center">
                  <svg className="w-4 h-4 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  <span className="text-gray-400">Starting analysis...</span>
                </div>
              )}
              
              {/* Show completed steps */}
              {sseMessages.slice(0, -1).map((message, index) => (
                <div key={index} className="flex items-center justify-center">
                  <svg className="w-4 h-4 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  <span className="text-gray-400">{message}</span>
                </div>
              ))}
              
              {/* Show current step */}
              {sseMessages.length > 0 && (
                <div className="flex items-center justify-center">
                  <div className="w-4 h-4 mr-2 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></div>
                  <span className="font-medium text-gray-800">{sseMessages[sseMessages.length - 1]}</span>
                </div>
              )}
            </div>
          </div>
          
          {uploadedFile && (
            <div className="text-sm text-gray-500 mt-4 font-mono">
              File size: {formatFileSize(uploadedFile.size)}
            </div>
          )}
          
          {analysisStartTime && (
            <div className="text-sm text-gray-500 font-mono">
              Time: {formatTime(elapsedTime)}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (appState === 'results') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-200">
        <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md flex flex-col items-center">
          <div className="text-green-600 mb-4">
            <svg className="w-16 h-16" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-gray-800 mb-6">Analysis Complete!</h1>
          <div className="text-gray-600 text-center mb-6">
            {analysisResult?.analysis && (
              <div className="mb-4">
                <p className="text-lg mb-2">
                  <span className="font-semibold">Predicted score:</span> {analysisResult.analysis.score}
                </p>
                <p className="text-lg">
                  <span className="font-semibold">Detected maneuvers:</span> {analysisResult.analysis.maneuvers.map(m => m.name.toLowerCase()).join(', ')}
                </p>
              </div>
            )}
            {analysisResult?.video_url && (
              <div className="mt-4 w-full">
                <video 
                  controls 
                  className="w-full rounded-lg shadow-md mb-6"
                  preload="metadata"
                >
                  <source src={analysisResult.video_url} type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
                <a 
                  href={analysisResult.video_url} 
                  download
                  className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-full text-base font-semibold hover:bg-blue-700 transition-colors"
                >
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Save Video
                </a>
              </div>
            )}
          </div>
          <button
            onClick={handleUploadAnother}
            className="px-6 py-3 bg-gray-600 text-white rounded-full text-base font-semibold hover:bg-gray-700 transition-colors"
          >
            Upload Another Video
          </button>
        </div>
      </div>
    );
  }

  if (appState === 'error') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-200">
        <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md flex flex-col items-center">
          <div className="text-red-600 mb-4">
            <svg className="w-16 h-16" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-gray-800 mb-6">Something went wrong</h1>
          <div className="text-gray-600 text-center mb-6">
            <p className="mb-4 text-lg">{error}</p>
          </div>
          <button
            onClick={handleUploadAnother}
            className="px-6 py-3 bg-gray-600 text-white rounded-full text-base font-semibold hover:bg-gray-700 transition-colors"
          >
            Upload Another Video
          </button>
        </div>
      </div>
    );
  }

  // Upload state (default)
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-200">
      <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md flex flex-col items-center">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">Upload a surf video</h1>
        
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileInputChange}
          className="hidden"
        />
        <button
          onClick={handleButtonClick}
          disabled={isUploading}
          className="mb-4 px-5 py-2 bg-white border border-gray-300 text-blue-600 rounded-full text-base font-semibold flex items-center justify-center gap-2 shadow-sm hover:bg-gray-50 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5-5m0 0l5 5m-5-5v12" />
          </svg>
          {isUploading ? 'Uploading...' : 'Select a file'}
        </button>
        {!isMobile && (
          <>
            <div className="text-gray-400 font-medium mb-4">or</div>
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              className={`w-full flex flex-col items-center justify-center border-2 border-dashed rounded-lg py-8 cursor-pointer transition ${
                isDraggingFiles 
                  ? 'border-blue-500 bg-blue-200' 
                  : 'border-blue-300 bg-blue-50'
              }`}
            >
              <span className="text-blue-500 font-semibold">Drag and drop a file here</span>
            </div>
          </>
        )}
        
        {!isMobile && (
          <div className="mt-4 text-xs text-gray-500 text-center">
            <p>Supported formats: MP4, MOV, and AVI</p>
            <p>Max file size: {MAX_FILE_SIZE_MB}MB</p>
          </div>
        )}
      </div>
    </div>
  );
}

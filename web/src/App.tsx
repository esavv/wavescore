import { useRef, useState } from "react";
import { UPLOAD_VIDEO_ENDPOINT } from "./constants";

type AppState = 'upload' | 'interim' | 'results' | 'error';

interface AnalysisResult {
  message: string;
  video_url?: string;
  analysis?: {
    score: number;
    maneuvers: Array<{name: string; start_time: number; end_time: number}>;
  };
}

export default function App() {
  const [appState, setAppState] = useState<AppState>('upload');
  const [sseMessages, setSseMessages] = useState<string[]>([]);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): string | null => {
    // Check file type
    if (!file.type.startsWith('video/')) {
      return 'Please select a video file (MP4, MOV, AVI, etc.)';
    }
    
    // Check file size (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      return 'File size must be less than 50MB';
    }
    
    return null;
  };

  const uploadVideo = async (file: File) => {
    setIsUploading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(UPLOAD_VIDEO_ENDPOINT, {
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
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleUploadAnother = () => {
    setAppState('upload');
    setSseMessages([]);
    setAnalysisResult(null);
    setError(null);
  };

  // Render different states
  if (appState === 'interim') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-200">
        <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md flex flex-col items-center">
          <h1 className="text-2xl font-bold text-gray-800 mb-6">Analyzing Video...</h1>
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
          <div className="text-gray-600 text-center mb-4">
            {sseMessages.length > 0 ? (
              <p className="font-medium">{sseMessages[sseMessages.length - 1]}</p>
            ) : (
              <p>Starting analysis...</p>
            )}
          </div>
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
        <div className="text-gray-400 font-medium mb-4">or</div>
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className="w-full flex flex-col items-center justify-center border-2 border-dashed border-blue-300 bg-blue-50 rounded-lg py-8 cursor-pointer hover:bg-blue-200 transition"
        >
          <span className="text-blue-500 font-semibold">Drag and drop a file here</span>
        </div>
        
        <div className="mt-4 text-xs text-gray-500 text-center">
          <p>Supported formats: MP4, MOV, AVI, and other video files</p>
          <p>Maximum file size: 50MB</p>
        </div>
      </div>
    </div>
  );
}

import { useRef } from "react";

export default function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    // TODO: handle dropped files
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-200">
      <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md flex flex-col items-center">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">Upload a surf video</h1>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="hidden"
        />
        <button
          onClick={handleButtonClick}
          className="mb-4 px-5 py-2 bg-white border border-gray-300 text-blue-600 rounded-full text-base font-semibold flex items-center justify-center gap-2 shadow-sm hover:bg-gray-50 transition"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5-5m0 0l5 5m-5-5v12" />
          </svg>
          Select a file
        </button>
        <div className="text-gray-400 font-medium mb-4">or</div>
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className="w-full flex flex-col items-center justify-center border-2 border-dashed border-blue-300 bg-blue-50 rounded-lg py-8 cursor-pointer hover:bg-blue-200 transition"
        >
          <span className="text-blue-500 font-semibold">Drag and drop a file here</span>
        </div>
      </div>
    </div>
  );
}

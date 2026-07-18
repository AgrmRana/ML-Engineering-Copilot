import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, Project } from '@/lib/api'
import { Upload as UploadIcon, CheckCircle, AlertCircle } from 'lucide-react'

export default function Upload() {
  const [selectedProject, setSelectedProject] = useState<number | null>(null)
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const queryClient = useQueryClient()

  const { data: projects } = useQuery({
    queryKey: ['projects'],
    queryFn: () => apiClient.listProjects().then((res) => res.data),
  })

  const uploadMutation = useMutation({
    mutationFn: async (files: FileList) => {
      if (!selectedProject) throw new Error('No project selected')
      
      const uploadPromises = Array.from(files).map((file) =>
        apiClient.uploadDocument(selectedProject, file).then((res) => res.data)
      )
      
      return Promise.all(uploadPromises)
    },
    onSuccess: () => {
      setUploadStatus('success')
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      setTimeout(() => {
        setUploadStatus('idle')
        setSelectedFiles(null)
      }, 2000)
    },
    onError: () => {
      setUploadStatus('error')
    },
  })

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedFiles(e.target.files)
  }

  const handleUpload = async () => {
    if (selectedFiles && selectedProject) {
      setUploadStatus('uploading')
      uploadMutation.mutate(selectedFiles)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Upload Documents</h1>
        <p className="mt-2 text-gray-600">Add documents to your projects for analysis</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Project Selection */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Select Project</h2>
          {!projects || projects.length === 0 ? (
            <p className="text-gray-500">No projects available. Create one first.</p>
          ) : (
            <div className="space-y-2">
              {projects.map((project) => (
                <button
                  key={project.id}
                  onClick={() => setSelectedProject(project.id)}
                  className={`w-full p-4 rounded-lg border text-left transition-colors ${
                    selectedProject === project.id
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium text-gray-900">{project.name}</div>
                  <div className="text-sm text-gray-600 mt-1">
                    {project.document_count || 0} documents
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* File Upload */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Upload Files</h2>
          <div className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-400 transition-colors">
              <UploadIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600 mb-2">
                Drag and drop files here, or click to select
              </p>
              <input
                type="file"
                multiple
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
                accept=".pdf,.md,.txt,.docx,.csv,.json,.py,.ipynb"
              />
              <label
                htmlFor="file-upload"
                className="btn btn-secondary cursor-pointer"
              >
                Select Files
              </label>
              <p className="text-xs text-gray-500 mt-2">
                Supported: PDF, Markdown, TXT, DOCX, CSV, JSON, Python, Jupyter
              </p>
            </div>

            {selectedFiles && selectedFiles.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm font-medium text-gray-700">
                  {selectedFiles.length} file(s) selected:
                </p>
                <div className="max-h-40 overflow-y-auto space-y-1">
                  {Array.from(selectedFiles).map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-2 bg-gray-50 rounded text-sm"
                    >
                      <span className="text-gray-700 truncate">{file.name}</span>
                      <span className="text-gray-500 text-xs">
                        {(file.size / 1024).toFixed(1)} KB
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedFiles && selectedFiles.length > 0 && selectedProject && uploadStatus !== 'uploading' && (
              <button
                onClick={handleUpload}
                className="btn btn-primary w-full"
                disabled={uploadStatus === 'success' || uploadStatus === 'error'}
              >
                {uploadStatus === 'success' ? 'Upload Complete' : 
                 uploadStatus === 'error' ? 'Upload Failed - Try Again' :
                 'Upload Documents'}
              </button>
            )}

            {uploadStatus === 'uploading' && (
              <div className="text-center py-4">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
                <p className="text-gray-600 mt-2">Uploading documents...</p>
              </div>
            )}

            {uploadStatus === 'success' && (
              <div className="flex items-center justify-center gap-2 text-green-600 py-4">
                <CheckCircle className="h-5 w-5" />
                <span>Documents uploaded successfully!</span>
              </div>
            )}

            {uploadStatus === 'error' && (
              <div className="flex items-center justify-center gap-2 text-red-600 py-4">
                <AlertCircle className="h-5 w-5" />
                <span>Upload failed. Please try again.</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

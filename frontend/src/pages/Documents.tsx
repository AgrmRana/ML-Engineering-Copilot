import { useQuery } from '@tanstack/react-query'
import { apiClient, Document } from '@/lib/api'
import { FileText, Trash2 } from 'lucide-react'
import { formatDate } from '@/lib/utils'

export default function Documents() {
  const { data: projects } = useQuery({
    queryKey: ['projects'],
    queryFn: () => apiClient.listProjects().then((res) => res.data),
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Documents</h1>
        <p className="mt-2 text-gray-600">View and manage uploaded documents</p>
      </div>

      {!projects || projects.length === 0 ? (
        <div className="card text-center py-12">
          <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 mb-4">No projects with documents yet</p>
          <p className="text-sm text-gray-400">Upload documents to a project first</p>
        </div>
      ) : (
        <div className="space-y-6">
          {projects.map((project) => {
            const { data: documents } = useQuery({
              queryKey: ['documents', project.id],
              queryFn: () => apiClient.listDocuments(project.id).then((res) => res.data),
            })

            if (!documents || documents.length === 0) return null

            return (
              <div key={project.id} className="card">
                <h2 className="text-lg font-semibold mb-4">{project.name}</h2>
                <div className="space-y-2">
                  {documents.map((doc) => (
                    <div
                      key={doc.id}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <FileText className="h-5 w-5 text-gray-400" />
                        <div>
                          <p className="font-medium text-gray-900">{doc.filename}</p>
                          <p className="text-sm text-gray-500">
                            {doc.file_type.toUpperCase()} • {doc.chunk_count} chunks
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-sm text-gray-500">
                          {formatDate(doc.created_at)}
                        </span>
                        <button className="text-gray-400 hover:text-red-600">
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

import { useQuery } from '@tanstack/react-query'
import { apiClient, Project } from '@/lib/api'
import { FolderOpen, FileText, MessageSquare, TrendingUp } from 'lucide-react'
import { Link } from 'react-router-dom'
import { formatNumber } from '@/lib/utils'

export default function Dashboard() {
  const { data: projects, isLoading } = useQuery({
    queryKey: ['projects'],
    queryFn: () => apiClient.listProjects().then((res) => res.data),
  })

  const stats = [
    { name: 'Total Projects', value: projects?.length || 0, icon: FolderOpen, color: 'bg-blue-500' },
    { name: 'Total Documents', value: projects?.reduce((acc, p) => acc + (p.document_count || 0), 0) || 0, icon: FileText, color: 'bg-green-500' },
    { name: 'Conversations', value: 0, icon: MessageSquare, color: 'bg-purple-500' },
    { name: 'Reports Generated', value: 0, icon: TrendingUp, color: 'bg-orange-500' },
  ]

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-2 text-gray-600">Overview of your ML workspace</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div key={stat.name} className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{stat.name}</p>
                <p className="mt-2 text-3xl font-bold text-gray-900">{formatNumber(stat.value)}</p>
              </div>
              <div className={`${stat.color} p-3 rounded-lg`}>
                <stat.icon className="h-6 w-6 text-white" />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Projects */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900">Recent Projects</h2>
          <Link to="/projects" className="text-primary-600 hover:text-primary-700 text-sm font-medium">
            View all →
          </Link>
        </div>

        {isLoading ? (
          <div className="text-center py-8 text-gray-500">Loading projects...</div>
        ) : projects && projects.length > 0 ? (
          <div className="space-y-4">
            {projects.slice(0, 5).map((project) => (
              <Link
                key={project.id}
                to={`/projects/${project.id}`}
                className="block p-4 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium text-gray-900">{project.name}</h3>
                    <p className="text-sm text-gray-600 mt-1">
                      {project.description || 'No description'}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">{project.document_count || 0} documents</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {new Date(project.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-gray-500 mb-4">No projects yet</p>
            <Link
              to="/upload"
              className="btn btn-primary inline-flex items-center gap-2"
            >
              Upload your first project
            </Link>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Link
            to="/upload"
            className="p-4 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors text-center"
          >
            <div className="text-2xl mb-2">📁</div>
            <p className="font-medium text-gray-900">Upload Project</p>
            <p className="text-sm text-gray-600 mt-1">Add documents to analyze</p>
          </Link>
          <Link
            to="/search"
            className="p-4 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors text-center"
          >
            <div className="text-2xl mb-2">🔍</div>
            <p className="font-medium text-gray-900">Search</p>
            <p className="text-sm text-gray-600 mt-1">Find information across projects</p>
          </Link>
          <Link
            to="/assistant"
            className="p-4 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors text-center"
          >
            <div className="text-2xl mb-2">🤖</div>
            <p className="font-medium text-gray-900">AI Assistant</p>
            <p className="text-sm text-gray-600 mt-1">Chat with your documents</p>
          </Link>
        </div>
      </div>
    </div>
  )
}

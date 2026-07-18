import { FileBarChart } from 'lucide-react'

export default function Reports() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Reports</h1>
        <p className="mt-2 text-gray-600">Generate and view AI-powered reports</p>
      </div>

      <div className="card text-center py-12">
        <FileBarChart className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-500 mb-4">Reports feature coming soon</p>
        <p className="text-sm text-gray-400">
          This feature will allow you to generate validation reports, 
          project summaries, and documentation automatically.
        </p>
      </div>
    </div>
  )
}

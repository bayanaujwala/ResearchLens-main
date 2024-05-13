require 'sinatra'
require 'json'
require 'anystyle'

set :bind, '0.0.0.0'
set :logging, false

post '/parse' do
  content_type :json
  body_content = request.body.read
  # request.body.rewind # This is typically necessary for Rack input, but we read first, then rewind for safety.

  body_content.force_encoding('UTF-8')

  if body_content.empty?
    status 400 # Bad Request if the body is empty
    return { error: 'Empty request body. Please provide text to parse.' }.to_json
  end

  begin
    result = AnyStyle.parse body_content
    result.to_json
  rescue => e
    status 500 # Internal Server Error
    { error: 'Failed to parse the provided text.', details: e.message }.to_json
  end
end


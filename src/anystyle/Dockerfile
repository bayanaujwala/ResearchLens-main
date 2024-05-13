FROM ruby:latest

# Install Sinatra and anystyle
RUN gem install sinatra anystyle rackup

# Copy the server script into the container
COPY server.rb /app/

# Set the working directory
WORKDIR /app

# Expose the port Sinatra will run on
EXPOSE 4567

# Start the Sinatra server
CMD ["ruby", "server.rb"]


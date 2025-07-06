#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#define STR_LITERAL_LABEL(x) #x 
#define STR_LITERAL_VALUE(y) STR_LITERAL_LABEL(y) 

int main(void) // argc, char *argv[])
{
    
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        perror("Pipe creation failed");
        fprintf(stderr, "Failed to create pipe: %s\n", strerror(errno));
        return 1;
    }
    printf("Pipe created successfully\n");

    // 0 is read end, 1 is write end
    // We can use these file descriptors to communicate with the child process
    const int read_end = 0;
    const int write_end = 1;

    const size_t width = 600; // Set the desired width 
    const size_t height = 600; // Set the desired height 
    const size_t fps = 60; // Set the desired frames per second 
    uint32_t pixels[width * height]; // Array to store the pixel data
    
    pid_t child_pid = fork();
    if (child_pid < 0) {
        perror("Fork failed");
        fprintf(stderr, "Failed to fork process: %s\n", strerror(errno));
        return 1;
    }
    if (child_pid == 0) {
    
        if (dup2(pipefd[read_end], STDIN_FILENO) < 0) {
            perror("dup2 failed");
            fprintf(stderr, "Failed to redirect stdin to pipe: %s\n", strerror(errno));
            return 1;   
        }
        close(pipefd[write_end]); // Close the write end of the pipe in the child process
        printf("close pipe --- child write end\n");
  
        // Child process
     
        int return_value = execlp("ffmpeg", 
            "ffmpeg",
            "-loglevel", "debug", // Set log level to debug
            "-f", "rawvideo", // Input format is raw video
            "-r", STR_LITERAL_VALUE(fps), // Set the frame rate to 60 fps
            "-s", STR_LITERAL_VALUE(width) "x" STR_LITERAL_VALUE(height), // Set the size of the video
            "-pix_fmt", "rgba", // Pixel format is RGBA
            "-y", // Overwrite output file without asking
            "-an",// Disable audio
            "-i", "-",
            "-c:v", "libx264",
            // Input from stdin (pipe)
            "output.mp4",
            //...
            NULL
        );
        if (return_value < 0) {
            perror("execlp failed");
            fprintf(stderr, "Failed to execute ffmpeg as child process: %s\n", strerror(errno));
            
            return 1;
        }
        printf("ffmpeg child process... started\n");
        return 0;
    }


    for (int i = 0; i < width * height; ++i) {
        // Simulate writing data to the pipe
        pixels[i] = 0xFF0000FF; // Red color in RGBA format
    }
    
    //ssize_t bytes_written = 0;
    size_t duration = 5; // Duration in seconds for the video
    printf("Parent process...writing to pipe for %zu seconds\n", duration);
    for (int i = 0; i < 60 * duration; ++i) {
        write(pipefd[write_end], pixels, width * height * sizeof(*pixels));
    }
    
    printf("Parent process...child pid: %d\n", child_pid);

    wait(NULL); // Wait for the child process to finish (return 0);
    printf("video rendered --- child process finished\n");

    close(pipefd[write_end]); // Close the write end of the pipe in the parent process
    printf("close pipe --- parent write end\n");
    return 0; // Return success

    // Note: The parent process should close the write end of the pipe after writing
    // and the child process should close the read end of the pipe after reading.
    // This ensures that the pipe is properly closed and no resources are leaked.
}
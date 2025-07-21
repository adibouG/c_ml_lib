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

int main(int argc, char *argv[]) // argc, char *argv[])
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
     #define read_end 0 // Set the desired frames per second 
    #define write_end 1 // Set the desired frames per second 
    #define fps 60 // Set the desired frames per second 
    #define width 600 // Set the desired frames per second 
    #define height 600 // Set the desired frames per second 
   
    // const int read_end = 0;
    //const int write_end = 1;
    //const int width = 600; // Set the desired width 
    //const int height = 600; // Set the desired height 
    //const int fps = 60; // Set the desired frames per second 
    uint32_t pixels[width * height]; // Array to store the pixel data
    int wstatus = -999; // Variable to store the status of the child process
    pid_t wait_status; 
       
    pid_t  child_pid = fork();
    if (child_pid < 0) {
        perror("Fork failed");
        fprintf(stderr, "Failed to fork process: %s\n", strerror(errno));
        return 1;
    }
    if (child_pid == 0) {
        // Child process
        printf("Child process... pid: %d\n", getpid());
        if (dup2(pipefd[read_end], STDIN_FILENO) < 0) {
            perror("dup2 failed");
            fprintf(stderr, "Failed to redirect stdin to pipe: %s\n", strerror(errno));
            return -1;   
        }
        close(pipefd[write_end]); // Close the write end of the pipe in the child process
        printf("close pipe --- child write end\n");
      
        // Child process
        char fps_str[16];
        snprintf(fps_str, sizeof(fps_str), "%d", fps);
        char res_str[32];
        snprintf(res_str, sizeof(res_str), "%dx%d", (int) width, (int) height);

        int return_value = execlp("ffmpeg", 
            "ffmpeg",
            "-loglevel", "debug", // Set log level to debug
            "-f", "rawvideo", // Input format is raw video
            "-r", fps_str, // Set the frame rate(fps, // Set the frame rate to 60 fps
            "-s", res_str, // Set the resolution(width) "x" STR_LITERAL_VALUE(height), // Set the size of the video
            "-pix_fmt", "rgba", // Pixel format is RGBA
            "-y", // Overwrite output file without asking
            "-an",// Disable audio
            //
            "-i", "-",  // Input from stdin (pipe)
            "-c:v", "libx264",
            //
            "output.mp4",
            //...
            NULL
        );
        if (return_value < 0) {
            perror("execlp failed");
            fprintf(stderr, "Failed to execute ffmpeg as child process: %s\n", strerror(errno));
            
            return return_value;
        }
        printf("ffmpeg child process... started\n");
        pause();
        printf("ffmpeg child process... ended\n");
        _exit(0);
        return 0; // return_value;
    }
    else {
        // Parent process
        printf("Parent process... pid: %d\n", getpid());
        close(pipefd[read_end]); // Close the read end of the pipe in the parent process
        printf("close pipe --- parent read end\n");

        // Fill the pixels array with some data (e.g., red color)
        // This is just an example; you can fill it with actual pixel data as needed
    for (int i = 0; i < width * height; ++i) {
        // Simulate writing data to the pipe
        pixels[i] = 0xFF0000FF; // Red color in RGBA format
    }
    

    
     //ssize_t bytes_written = 0;
    size_t duration = 7; // Duration in seconds for the video
    printf("Parent process...writing to pipe for %zu seconds\n", duration);
    for (int i = 0; i < 60 * duration; ++i) {
        write(pipefd[write_end], pixels, width * height * sizeof(*pixels));
    }
    
    close(pipefd[write_end]); // Close the write end of the pipe in the parent process
    printf("close pipe --- parent write end\n");
    
    do  {
        printf("Parent process...child pid: %d\n", child_pid);
        wait_status = waitpid(child_pid, &wstatus, WUNTRACED | WCONTINUED ); // Wait for the child to finish;
        if (wait_status < 0) {
            perror("waitpid failed");
            fprintf(stderr, "Failed to wait for child process: %s\n", strerror(errno));
            return 1;
        }
        if (WIFEXITED(wstatus)) {
            printf("Child process exited with status %d\n", WEXITSTATUS(wstatus));
        } else if (WIFSIGNALED(wstatus)) {
            printf("Child process terminated by signal %d\n", WTERMSIG(wstatus));
        } else if (WIFSTOPPED(wstatus)) {
            printf("Child process stopped by signal %d\n", WSTOPSIG(wstatus));
        } else if (WIFCONTINUED(wstatus)) {
            printf("Child process continued\n");
        }
    } while (!WIFEXITED(wstatus) && !WIFSIGNALED(wstatus));
        
    printf("video rendered --- child process finished\n");
    exit(0);    
        return 0;
    //return (EXIT_SUCCESS); // return 0; // Return success
}
    // Note: The parent process should close the write end of the pipe after writing
    // and the child process should close the read end of the pipe after reading.
    // This ensures that the pipe is properly closed and no resources are leaked.
}
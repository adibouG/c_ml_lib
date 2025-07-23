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
    
    int cmd_width = width; // Set the desired width 
    int cmd_height = height; // Set the desired height 
    int cmd_fps = fps; // Set the desired frames per second 
    
    if (argc > 1) {
        // If command line arguments are provided, use them to set width, height, and fps
        cmd_width = atoi(argv[1]);
        cmd_height = atoi(argv[2]);
        cmd_fps = atoi(argv[3]);
    }
    // const int read_end = 0;
    //const int write_end = 1;
    uint32_t pixels[width * height]; // Array to store the pixel data
    int wstatus = -999; // Variable to store the status of the child process
    pid_t wait_status; 
       
    pid_t  child_pid = fork();
    if (child_pid < 0) {
        perror("Fork failed");
        fprintf(stderr, "Failed to fork process: %s\n", strerror(errno));
        exit(1);
    }
    if (child_pid == 0) {
        // Child process
        printf("Child process... pid: %d\n", getpid());
        close(pipefd[write_end]); // Close the write end of the pipe in the child process
        printf("close pipe --- child write end\n");
        if (dup2(pipefd[read_end], STDIN_FILENO) < 0) {
            perror("dup2 failed");
            fprintf(stderr, "Failed to redirect stdin to pipe: %s\n", strerror(errno));
            exit(1);   
        }
       
     
      
        // Child process
        char fps_str[16];
        size_t duration = 7; // Duration in seconds for the video
        snprintf(fps_str, sizeof(fps_str), "%d", cmd_fps); // Convert fps to string
        char res_str[32];
        snprintf(res_str, sizeof(res_str), "%dx%d", (int) cmd_width, (int) cmd_height);

        int return_value = execlp("ffmpeg", 
            "ffmpeg",
            "-loglevel", "warning", // Set log level to debug
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
            close(pipefd[read_end]);
            _exit(1); // Exit the child process with an error code
        }
        else {
            printf("Child process...execlp returned successfully\n");
            close(pipefd[read_end]);
            _exit(0);
        }
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
    
        ssize_t bytes_written = 0;
        unsigned int duration = 10; // Duration in seconds for the video
        unsigned int frame_i = 0; // Frame counter for 
        printf("Parent process...writing to pipe for %zu seconds\n", duration);
        for (frame_i = 0 ; frame_i < cmd_fps * duration; ++frame_i) {
            ssize_t bits2write = width * height * sizeof(*pixels);
            write(pipefd[write_end], pixels, bits2write);
            bytes_written += bits2write / 8;
            printf("Parent process...added %zd bits to pipe, wrote %zd bytes to pipe\n",bits2write, bytes_written);
        }
        
        close(pipefd[write_end]); // Close the write end of the pipe in the parent process
        printf("close pipe --- parent write end\n");
       
        printf("Parent process...child pid: %d\n", child_pid);
        wait_status = waitpid(child_pid, &wstatus, WUNTRACED | WCONTINUED ); // Wait for the child to finish;
        if (wait_status < 0) {
            perror("waitpid failed");
            fprintf(stderr, "Failed to wait for child process: %s\n", strerror(errno));
            return 1;
        }
        printf("Parent process...waitpid returned %d\n", wait_status);
        
        // This block is here if you want to handle the child process status in a loop
        // For example, if you want to handle signals or check if the child is stopped or 
        // continued, you can uncomment this block and use this loop.
        /*  
        do {
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
        */
        printf("Parent process...child process finished with status %d\n", wstatus);
        printf("video rendered --- child process finished\n");
        printf("video rendered --- parent process finished\n");
        exit(0);    
    }
    // Note: The parent process should close the write end of the pipe after writing
    // and the child process should close the read end of the pipe after reading.
    // This ensures that the pipe is properly closed and no resources are leaked.
}
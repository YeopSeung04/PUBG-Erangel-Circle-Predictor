using System.Diagnostics;
using System.Runtime.InteropServices;

internal static class Program
{
    private const int SwRestore = 9;

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBoxW(IntPtr hWnd, string text, string caption, uint type);

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetConsoleWindow();

    [DllImport("user32.dll")]
    private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    private static int Main(string[] args)
    {
        var projectRoot = FindProjectRoot(AppContext.BaseDirectory);
        if (projectRoot is null)
        {
            ShowMessage("Circle Train", "Could not find Circle Train project root.");
            return 1;
        }

        var pythonArgs = args.Length > 0
            ? string.Join(" ", args.Select(Quote))
            : "--target-new 1000 --days 14 --shards steam,kakao --min-circles 2 --no-notify";

        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "py",
                Arguments = $"-m circle_train.auto_collect {pythonArgs}",
                WorkingDirectory = projectRoot,
                UseShellExecute = false,
                RedirectStandardOutput = false,
                RedirectStandardError = false,
            }
        };

        try
        {
            ShowWindow(GetConsoleWindow(), SwRestore);
            process.Start();
            process.WaitForExit();

            var status = process.ExitCode == 0 ? "finished" : $"finished with errors ({process.ExitCode})";
            ShowMessage("Circle Train", $"Auto collection {status}.\n\nProject: {projectRoot}");
            return process.ExitCode;
        }
        catch (Exception ex)
        {
            ShowMessage("Circle Train", $"Auto collection failed.\n\n{ex.Message}");
            return 1;
        }
    }

    private static string? FindProjectRoot(string startDirectory)
    {
        var directory = new DirectoryInfo(startDirectory);
        while (directory is not null)
        {
            var pyproject = Path.Combine(directory.FullName, "pyproject.toml");
            var sourceDirectory = Path.Combine(directory.FullName, "src", "circle_train");
            if (File.Exists(pyproject) && Directory.Exists(sourceDirectory))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    private static string Quote(string value)
    {
        if (!value.Any(char.IsWhiteSpace))
        {
            return value;
        }

        return $"\"{value.Replace("\"", "\\\"")}\"";
    }

    private static void ShowMessage(string title, string message)
    {
        MessageBoxW(IntPtr.Zero, message, title, 0x00001000);
    }
}

import * as vscode from 'vscode';
import * as path from 'path';

export function activate(context: vscode.ExtensionContext) {
    const factory = new ConcreteDebugAdapterFactory(context);
    context.subscriptions.push(
        vscode.debug.registerDebugAdapterDescriptorFactory('concrete', factory)
    );
}

export function deactivate() {
    // Nothing to clean up.
}

class ConcreteDebugAdapterFactory implements vscode.DebugAdapterDescriptorFactory {
    constructor(private readonly context: vscode.ExtensionContext) {}

    createDebugAdapterDescriptor(
        session: vscode.DebugSession,
        _executable: vscode.DebugAdapterExecutable | undefined
    ): vscode.ProviderResult<vscode.DebugAdapterDescriptor> {
        const config = session.configuration;
        const pythonPath: string = config.pythonPath || 'python3';

        // The DAP server Python script is bundled at dap-server/ inside the extension
        const serverScript = path.join(
            this.context.extensionPath,
            'dap-server',
            'concrete_dap_server.py'
        );

        return new vscode.DebugAdapterExecutable(pythonPath, [serverScript]);
    }
}

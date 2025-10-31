from pipeline.types.metrics import Patcher
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
import unittest


class TestPatcher(unittest.TestCase):
    @unittest.skip("Skipping test_apply_patch_training")
    def test_apply_patch_training(self):
        project = "simplelocalize-cli"
        log_path = None
        container_path = "/home/xchen6/breaking_updates_rl/data/dataset/0a11c04038eae517540051dbf51f7f26b7221f20/0a11c04038eae517540051dbf51f7f26b7221f20.sif"
        # host_project_path = "/home/xchen6/breaking_updates_rl/data/dataset/0a11c04038eae517540051dbf51f7f26b7221f20/simplelocalize-cli"
        patcher = Patcher(project, container_path, log_path, [])
        
        patch = "package io.simplelocalize.cli.configuration;\n\nimport io.simplelocalize.cli.exception.ConfigurationException;\nimport org.slf4j.Logger;\nimport org.slf4j.LoggerFactory;\nimport org.yaml.snakeyaml.Yaml;\nimport org.yaml.snakeyaml.constructor.Constructor;\n\nimport java.io.File;\nimport java.io.FileInputStream;\nimport java.io.FileNotFoundException;\nimport java.io.InputStream;\nimport java.net.URLDecoder;\nimport java.nio.charset.StandardCharsets;\nimport java.nio.file.Path;\n\npublic final class ConfigurationLoader\n{\n\n  private static final Path DEFAULT_CONFIG_FILE_NAME = Path.of(\"simplelocalize.yml\");\n\n  private final Logger log = LoggerFactory.getLogger(ConfigurationLoader.class);\n\n  public Configuration loadOrGetDefault(Path configurationFilePath)\n  {\n    ConfigurationLoader configurationLoader = new ConfigurationLoader();\n\n    if (configurationFilePath == null)\n    {\n      configurationFilePath = DEFAULT_CONFIG_FILE_NAME;\n    }\n\n    return configurationLoader.load(configurationFilePath);\n  }\n\n  private Configuration load(Path configurationFilePath)\n  {\n    File file = new File(URLDecoder.decode(String.valueOf(configurationFilePath.toFile()), StandardCharsets.UTF_8));\n    Constructor yamlTargetClass = new Constructor<>();\n    Yaml yaml = new Yaml(yamlTargetClass);\n\n    log.info(\"Loading configuration file from: {}\", configurationFilePath);\n    Configuration configuration;\n    try\n    {\n      InputStream inputStream = new FileInputStream(file);\n      configuration = yaml.load(inputStream);\n      log.info(\"Configuration file loaded successfully\");\n    } catch (FileNotFoundException e)\n    {\n      log.info(\"Configuration file not present\");\n      return new Configuration();\n    } catch (Exception e)\n    {\n      log.error(\"Unable to load configuration: {}\", e.getMessage());\n      throw new ConfigurationException();\n    }\n    return configuration;\n\n  }\n\n}\n"
        container_file = "/simplelocalize-cli/src/main/java/io/simplelocalize/cli/configuration/ConfigurationLoader.java"
        
        error_log, success = patcher.apply_patch_training(patch, container_file)
        
        print("Error Log:", error_log.to_jsonable())
        print("Build Success:", success)
        
        ground_truth_log_path = "/home/xchen6/breaking_updates_rl/results/google/gemma-3-12b-it_20251007-173850/0a11c04038eae517540051dbf51f7f26b7221f20/0a11c04038eae517540051dbf51f7f26b7221f20.log"
        ground_truth_error_log = MavenErrorLog.from_file(ground_truth_log_path, MavenErrorParser())
        print("Ground Truth Error Log:", ground_truth_error_log.to_jsonable())
        self.assertEqual(error_log.to_jsonable(), ground_truth_error_log.to_jsonable())

    @unittest.skip("Skipping test_apply_patch_training")
    def test_apply_patch_training_mem_overflow(self):
        project = "quickfixj"
        log_path = None
        container_path = "/home/xchen6/breaking_updates_rl/data/dataset/00a7cc31784ac4a9cc27d506a73ae589d6df36d6/00a7cc31784ac4a9cc27d506a73ae589d6df36d6.sif"
        # host_project_path = "/home/xchen6/breaking_updates_rl/data/dataset/00a7cc31784ac4a9cc27d506a73ae589d6df36d6/quickfixj"
        patcher = Patcher(project, container_path, log_path, [])
        
        patch = "/*******************************************************************************\n * Copyright (c) quickfixengine.org  All rights reserved.\n *\n * This file is part of the QuickFIX FIX Engine\n *\n * This file may be distributed under the terms of the quickfixengine.org\n * license as defined by quickfixengine.org and appearing in the file\n * LICENSE included in the packaging of this file.\n *\n * This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING\n * THE WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A\n * PARTICULAR PURPOSE.\n *\n * See http://www.quickfixengine.org/LICENSE for licensing information.\n *\n * Contact ask@quickfixengine.org if any conditions of this licensing\n * are not clear to you.\n ******************************************************************************/\n\npackage quickfix.mina.ssl;\n\nimport java.net.InetSocketAddress;\nimport java.net.SocketAddress;\nimport javax.net.ssl.SSLContext;\n\nimport javax.net.ssl.SSLException;\nimport org.apache.mina.core.filterchain.IoFilterChain;\nimport org.apache.mina.core.session.IoSession;\nimport org.apache.mina.filter.ssl.SslFilter;\nimport org.slf4j.Logger;\nimport org.slf4j.LoggerFactory;\n\n/**\n * An extended SSL filter based on MINA {@link SslFilter} that applies\n * some adaptations.\n */\npublic class SSLFilter extends SslFilter {\n\n    private final Logger log = LoggerFactory.getLogger(getClass());\n    private boolean useSNI;\n\n    public SSLFilter(SSLContext sslContext, boolean autoStart) {\n        super(sslContext, autoStart);\n    }\n\n    public SSLFilter(SSLContext sslContext) {\n        super(sslContext);\n    }\n\n    /**\n     * Called from {@link SslFilter#onPreAdd} every time a new\n     * session is created which makes it impossible to override enabled cipher\n     * suites configuration.\n     */\n    public void setEnabledCipherSuites(String[] cipherSuites) {\n    }\n\n    public void setCipherSuites(String[] cipherSuites) {\n        super.setEnabledCipherSuites(cipherSuites);\n    }\n\n    /**\n     * Called before filter is added into the chain.\n     * We activate Server Name Indication if it is enabled in the session config.\n     */\n    public void onPreAdd(IoFilterChain parent, String name, NextFilter nextFilter)\n        throws SSLException {\n\n        if (useSNI) {\n            IoSession session = parent.getSession();\n            SocketAddress remoteAddress = session.getRemoteAddress();\n\n            if (remoteAddress instanceof InetSocketAddress) {\n                // activate the SNI support in the JSSE SSLEngine\n                log.info(\"Activating TLS SNI support for peer address: {}\", remoteAddress);\n                //session.setAttribute(PEER_ADDRESS, remoteAddress);\n            }\n        }\n\n        super.onPreAdd(parent, name, nextFilter);\n    }\n\n    public void setUseSNI(boolean useSNI) {\n        this.useSNI = useSNI;\n    }\n}\n"
        container_file = "/quickfixj/quickfixj-core/src/main/java/quickfix/mina/initiator/IoSessionInitiator.java"
        
        error_log, success = patcher.apply_patch_training(patch, container_file)
        
        print("Error Log:", error_log.to_jsonable())
        print("Build Success:", success)
        
        ground_truth_log_path = "/home/xchen6/breaking_updates_rl/results/google/gemma-3-12b-it_20251007-173850/00a7cc31784ac4a9cc27d506a73ae589d6df36d6/00a7cc31784ac4a9cc27d506a73ae589d6df36d6.log"
        ground_truth_error_log = MavenErrorLog.from_file(ground_truth_log_path, MavenErrorParser())
        print("Ground Truth Error Log:", ground_truth_error_log.to_jsonable())
        self.assertEqual(error_log.to_jsonable(), ground_truth_error_log.to_jsonable())
        
        
    def test_Xchange_time_cost(self):
        project = "XChange"
        log_path = None
        container_path = "/home/xchen6/breaking_updates_rl/data/dataset/00a7cc31784ac4a9cc27d506a73ae589d6df36d6/00a7cc31784ac4a9cc27d506a73ae589d6df36d6.sif"
        # host_project_path = "/home/xchen6/breaking_updates_rl/data/dataset/00a7cc31784ac4a9cc27d506a73ae589d6df36d6/quickfixj"
        patcher = Patcher(project, container_path, log_path, [])
        
        patch = "/*******************************************************************************\n * Copyright (c) quickfixengine.org  All rights reserved.\n *\n * This file is part of the QuickFIX FIX Engine\n *\n * This file may be distributed under the terms of the quickfixengine.org\n * license as defined by quickfixengine.org and appearing in the file\n * LICENSE included in the packaging of this file.\n *\n * This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING\n * THE WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A\n * PARTICULAR PURPOSE.\n *\n * See http://www.quickfixengine.org/LICENSE for licensing information.\n *\n * Contact ask@quickfixengine.org if any conditions of this licensing\n * are not clear to you.\n ******************************************************************************/\n\npackage quickfix.mina.ssl;\n\nimport java.net.InetSocketAddress;\nimport java.net.SocketAddress;\nimport javax.net.ssl.SSLContext;\n\nimport javax.net.ssl.SSLException;\nimport org.apache.mina.core.filterchain.IoFilterChain;\nimport org.apache.mina.core.session.IoSession;\nimport org.apache.mina.filter.ssl.SslFilter;\nimport org.slf4j.Logger;\nimport org.slf4j.LoggerFactory;\n\n/**\n * An extended SSL filter based on MINA {@link SslFilter} that applies\n * some adaptations.\n */\npublic class SSLFilter extends SslFilter {\n\n    private final Logger log = LoggerFactory.getLogger(getClass());\n    private boolean useSNI;\n\n    public SSLFilter(SSLContext sslContext, boolean autoStart) {\n        super(sslContext, autoStart);\n    }\n\n    public SSLFilter(SSLContext sslContext) {\n        super(sslContext);\n    }\n\n    /**\n     * Called from {@link SslFilter#onPreAdd} every time a new\n     * session is created which makes it impossible to override enabled cipher\n     * suites configuration.\n     */\n    public void setEnabledCipherSuites(String[] cipherSuites) {\n    }\n\n    public void setCipherSuites(String[] cipherSuites) {\n        super.setEnabledCipherSuites(cipherSuites);\n    }\n\n    /**\n     * Called before filter is added into the chain.\n     * We activate Server Name Indication if it is enabled in the session config.\n     */\n    public void onPreAdd(IoFilterChain parent, String name, NextFilter nextFilter)\n        throws SSLException {\n\n        if (useSNI) {\n            IoSession session = parent.getSession();\n            SocketAddress remoteAddress = session.getRemoteAddress();\n\n            if (remoteAddress instanceof InetSocketAddress) {\n                // activate the SNI support in the JSSE SSLEngine\n                log.info(\"Activating TLS SNI support for peer address: {}\", remoteAddress);\n                //session.setAttribute(PEER_ADDRESS, remoteAddress);\n            }\n        }\n\n        super.onPreAdd(parent, name, nextFilter);\n    }\n\n    public void setUseSNI(boolean useSNI) {\n        this.useSNI = useSNI;\n    }\n}\n"
        container_file = "/quickfixj/quickfixj-core/src/main/java/quickfix/mina/initiator/IoSessionInitiator.java"
        
        error_log, success = patcher.apply_patch_training(patch, container_file)
        
        print("Error Log:", error_log.to_jsonable())
        print("Build Success:", success)
        
        ground_truth_log_path = "/home/xchen6/breaking_updates_rl/results/google/gemma-3-12b-it_20251007-173850/00a7cc31784ac4a9cc27d506a73ae589d6df36d6/00a7cc31784ac4a9cc27d506a73ae589d6df36d6.log"
        ground_truth_error_log = MavenErrorLog.from_file(ground_truth_log_path, MavenErrorParser())
        print("Ground Truth Error Log:", ground_truth_error_log.to_jsonable())
        self.assertEqual(error_log.to_jsonable(), ground_truth_error_log.to_jsonable())
        
if __name__ == "__main__":
    unittest.main()